//! Machine lifecycle handlers.
//!
//! These handlers manage persistent machines via the shared database,
//! accessible to both API and CLI commands.
//!
//! ## Limitations
//!
//! ### Name Length Limit
//!
//! Machine name length is bounded by the kernel's `sockaddr_un.sun_path`
//! limit (104 bytes on macOS, 108 on Linux). The full socket path is:
//!
//! ```text
//! ~/Library/Caches/smolvm/vms/{name}/agent.sock
//! ```
//!
//! Maximum usable name length therefore depends on the user's home directory.
//! For a typical macOS home (`/Users/<username>/`, ~20 chars), names can be
//! 50+ characters. The actual socket path is validated at create time via
//! [`crate::data::validate_socket_path_fits`] so overly-long names are
//! rejected with a clear error up front.
//!
//! Recommended: keep names short and descriptive (e.g., "dev-vm", "test-1").

use axum::{
    extract::{Path, Query, State},
    response::sse::{Event, KeepAlive, Sse},
    Json,
};
use std::convert::Infallible;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use crate::agent::{AgentClient, AgentManager, HostMount};
use crate::api::error::ApiError;
use crate::api::state::{vm_resources_to_spec, ApiState, MachineEntry};
use crate::api::types::{
    ApiErrorResponse, CreateMachineRequest, DataDirResponse, DeleteQuery, DeleteResponse, EnvVar,
    ForkMachineRequest, ListMachinesResponse, MachineInfo, MonitorQuery, MountInfo, MountSpec,
    NetworkTestRequest, NetworkTestResponse, PortSpec, ResizeMachineRequest, StartMachineRequest,
    UpdateMachineRequest,
};
use crate::config::{RecordState, RestartConfig, VmRecord};
use crate::machine::{
    CreateMachine, DataDirMachine, DeleteMachine, ForkMachine, GetMachine, ListMachines,
    LocalMachineService, MachineService, MachineStatus, MonitorEvent, MonitorMachine,
    NetworkTestMachine, StartMachine, StopMachine, UpdateMachine,
};
use crate::process::{
    is_alive, is_our_process_strict, stop_vm_process, VM_SIGKILL_TIMEOUT, VM_SIGTERM_TIMEOUT,
};
use crate::util::generate_machine_name;

/// Convert a typed core machine status to MachineInfo (pure mapping, no I/O).
fn machine_status_to_info(status: &MachineStatus) -> MachineInfo {
    let record = &status.record;
    let pid = if status.state == RecordState::Stopped {
        None
    } else {
        record.pid
    };
    MachineInfo {
        name: status.name.clone(),
        state: status.state.to_string(),
        cpus: record.cpus,
        mem: record.mem,
        pid,
        mounts: record
            .mounts
            .iter()
            .enumerate()
            .map(|(i, (source, target, readonly))| MountInfo {
                tag: HostMount::mount_tag(i),
                source: source.clone(),
                target: target.clone(),
                readonly: *readonly,
            })
            .collect(),
        ports: record
            .ports
            .iter()
            .map(|(host, guest)| PortSpec {
                host: *host,
                guest: *guest,
            })
            .collect(),
        network: record.network,
        storage_gb: record.storage_gb,
        overlay_gb: record.overlay_gb,
        created_at: record.created_at,
    }
}

/// Build a MachineEntry from a VmRecord and AgentManager.
///
/// Used by `start_machine` to register a machine in ApiState after boot
/// or during registry repair. Centralizes the record→entry conversion
/// so the two branches don't drift.
fn machine_entry_from_record(record: &VmRecord, manager: AgentManager) -> MachineEntry {
    let mounts = record
        .mounts
        .iter()
        .map(|(s, t, ro)| MountSpec {
            source: s.clone(),
            target: t.clone(),
            readonly: *ro,
        })
        .collect();
    let ports = record
        .ports
        .iter()
        .map(|(h, g)| PortSpec {
            host: *h,
            guest: *g,
        })
        .collect();
    MachineEntry {
        manager,
        mounts,
        ports,
        resources: vm_resources_to_spec(record.vm_resources()),
        restart: record.restart.clone(),
        network: record.network,
        secret_refs: record.secret_refs.clone(),
        source_smolmachine: record.source_smolmachine.clone(),
    }
}

/// Attempt graceful shutdown, then force-terminate if still running.
///
/// Uses verified signals to prevent killing an unrelated process if the
/// PID was recycled by the OS. Returns true if the process is confirmed
/// dead (or was never running), false if it may still be alive.
fn shutdown_machine_process(name: &str, pid: Option<i32>, pid_start_time: Option<u64>) -> bool {
    // Try graceful shutdown via vsock first.
    // If vsock connects, this confirms the process is our VM (identity verification).
    let manager = AgentManager::for_vm(name).ok();
    let mut vsock_confirmed = false;
    if let Some(ref manager) = manager {
        if let Ok(mut client) = AgentClient::connect(manager.vsock_socket()) {
            vsock_confirmed = true;
            let _ = client.shutdown();
        }
    }

    // PID-based signal handling.
    if let Some(pid) = pid {
        // Identity check: vsock acknowledgement OR strict PID start-time match.
        // We intentionally do NOT use the lenient is_our_process() here because
        // it treats any alive PID as "ours" when start_time is None — which risks
        // killing an unrelated process if the OS reused the PID.
        let identity_ok = vsock_confirmed || is_our_process_strict(pid, pid_start_time);

        if identity_ok {
            let _ = stop_vm_process(pid, VM_SIGTERM_TIMEOUT, VM_SIGKILL_TIMEOUT);
        } else {
            tracing::debug!(pid, name, "PID already dead");
        }

        // Post-check: verify the process is actually gone.
        if is_alive(pid) {
            tracing::warn!(pid, name, "process still alive after shutdown attempts");
            return false;
        }
    } else {
        // No PID available — check if VM is still reachable via vsock.
        if let Some(ref manager) = manager {
            if let Ok(mut client) = AgentClient::connect(manager.vsock_socket()) {
                if client.ping().is_ok() {
                    tracing::warn!(name, "VM still reachable via vsock but no PID to signal");
                    return false;
                }
            }
        }
    }

    true
}

/// Create a new machine.
#[utoipa::path(
    post,
    path = "/api/v1/machines",
    tag = "Machines",
    request_body = CreateMachineRequest,
    responses(
        (status = 200, description = "Machine created", body = MachineInfo),
        (status = 400, description = "Invalid request", body = ApiErrorResponse),
        (status = 409, description = "Machine already exists", body = ApiErrorResponse)
    )
)]
pub async fn create_machine(
    State(state): State<Arc<ApiState>>,
    Json(req): Json<CreateMachineRequest>,
) -> Result<Json<MachineInfo>, ApiError> {
    let source_count = [
        req.registry_ref.is_some(),
        req.from.is_some(),
        req.image.is_some(),
    ]
    .iter()
    .filter(|&&b| b)
    .count();
    if source_count > 1 {
        return Err(ApiError::BadRequest(
            "'registryRef', 'from', and 'image' are mutually exclusive".to_string(),
        ));
    }

    let mut req = req;
    if let Some(ref registry_ref) = req.registry_ref.clone() {
        let pulled_path =
            pull_from_registry(registry_ref, req.registry_identity_token.as_deref()).await?;
        req.from = Some(pulled_path);
        req.registry_ref = None;
    }

    let name = req.name.clone().unwrap_or_else(generate_machine_name);

    let (
        image,
        source_smolmachine,
        entrypoint,
        cmd,
        manifest_env,
        workdir,
        manifest_cpus,
        manifest_mem,
        manifest_net,
        manifest_secret_refs,
    ) = if let Some(ref sidecar_path) = req.from {
        let path = std::path::Path::new(sidecar_path);
        if !path.exists() {
            return Err(ApiError::BadRequest(format!(
                "sidecar file not found: {}",
                sidecar_path
            )));
        }
        let manifest = smolvm_pack::packer::read_manifest_from_sidecar(path)
            .map_err(|e| ApiError::internal(format!("read .smolmachine: {}", e)))?;
        let canonical = path
            .canonicalize()
            .unwrap_or_else(|_| path.to_path_buf())
            .to_string_lossy()
            .into_owned();
        for (key, secret_ref) in &manifest.secret_refs {
            crate::secrets::validate_ref(secret_ref, crate::secrets::ResolutionScope::Untrusted)
                .map_err(|e| {
                    ApiError::BadRequest(format!(
                        "packed secret '{}': {} (packs may not carry secret refs)",
                        key, e
                    ))
                })?;
        }
        (
            Some(manifest.image),
            Some(canonical),
            manifest.entrypoint,
            manifest.cmd,
            manifest.env,
            manifest.workdir,
            manifest.cpus,
            manifest.mem,
            manifest.network,
            manifest.secret_refs,
        )
    } else {
        (
            req.image.clone(),
            None,
            vec![],
            vec![],
            vec![],
            None,
            crate::data::resources::DEFAULT_MICROVM_CPU_COUNT,
            crate::data::resources::DEFAULT_MICROVM_MEMORY_MIB,
            req.network,
            Default::default(),
        )
    };

    crate::api::handlers::validate_request_secrets(&req.secrets)?;
    let (cpus, mem) = resolve_create_resources(&req, manifest_cpus, manifest_mem);
    let network = req.network || manifest_net;
    let restart = match req.restart {
        Some(ref spec) => {
            let policy = spec
                .policy
                .as_deref()
                .unwrap_or("never")
                .parse()
                .map_err(|e: String| ApiError::BadRequest(e))?;
            RestartConfig {
                policy,
                max_retries: spec.max_retries.unwrap_or(0),
                ..Default::default()
            }
        }
        None => RestartConfig::default(),
    };

    let mut create = CreateMachine::new(name.clone());
    create.image = image;
    create.source_smolmachine = source_smolmachine;
    create.entrypoint = entrypoint;
    create.cmd = cmd;
    create.cpus = cpus;
    create.memory_mib = mem;
    create.mounts = req
        .mounts
        .iter()
        .map(HostMount::try_from)
        .collect::<crate::Result<Vec<_>>>()
        .map_err(|e| ApiError::BadRequest(e.to_string()))?;
    create.ports = req
        .ports
        .iter()
        .map(crate::agent::PortMapping::from)
        .collect();
    create.net = network;
    create.env = manifest_env;
    create.workdir = workdir;
    create.storage_gb = req.storage_gb;
    create.overlay_gb = req.overlay_gb;
    create.allowed_cidrs = req.allowed_cidrs.clone();
    create.restart_policy = Some(restart.policy.clone());
    create.restart_max_retries = Some(restart.max_retries);
    create.restart_max_backoff_secs = Some(restart.max_backoff_secs);
    create.gpu = req.gpu;
    create.secret_refs = {
        let mut refs = manifest_secret_refs;
        refs.extend(req.secrets.clone());
        refs
    };

    let db = state.db().clone();
    let status =
        tokio::task::spawn_blocking(move || LocalMachineService::with_db(db).create(create))
            .await
            .map_err(|e| ApiError::internal(format!("task error: {}", e)))?
            .map_err(ApiError::from)?;

    let manager = tokio::task::spawn_blocking({
        let name = name.clone();
        let storage_gb = status.record.storage_gb;
        let overlay_gb = status.record.overlay_gb;
        move || AgentManager::for_vm_with_sizes(&name, storage_gb, overlay_gb)
    })
    .await
    .map_err(|e| ApiError::internal(format!("task error: {}", e)))?
    .map_err(|e| ApiError::internal(format!("failed to create agent manager: {}", e)))?;

    state.insert_machine(&name, machine_entry_from_record(&status.record, manager));

    Ok(Json(machine_status_to_info(&status)))
}

/// List all machines.
#[utoipa::path(
    get,
    path = "/api/v1/machines",
    tag = "Machines",
    responses(
        (status = 200, description = "List of machines", body = ListMachinesResponse),
        (status = 500, description = "Database error", body = ApiErrorResponse)
    )
)]
pub async fn list_machines(
    State(state): State<Arc<ApiState>>,
) -> Result<Json<ListMachinesResponse>, ApiError> {
    let db = state.db().clone();
    let statuses =
        tokio::task::spawn_blocking(move || LocalMachineService::with_db(db).list(ListMachines))
            .await
            .map_err(|e| ApiError::internal(format!("task error: {}", e)))?
            .map_err(ApiError::from)?;

    let machines = statuses.iter().map(machine_status_to_info).collect();

    Ok(Json(ListMachinesResponse { machines }))
}

/// Get machine status.
#[utoipa::path(
    get,
    path = "/api/v1/machines/{name}",
    tag = "Machines",
    params(
        ("name" = String, Path, description = "Machine name")
    ),
    responses(
        (status = 200, description = "Machine details", body = MachineInfo),
        (status = 404, description = "Machine not found", body = ApiErrorResponse)
    )
)]
pub async fn get_machine(
    State(state): State<Arc<ApiState>>,
    Path(name): Path<String>,
) -> Result<Json<MachineInfo>, ApiError> {
    let db = state.db().clone();
    let status = tokio::task::spawn_blocking({
        let name = name.clone();
        move || LocalMachineService::with_db(db).status(GetMachine::new(name))
    })
    .await
    .map_err(|e| ApiError::internal(format!("task error: {}", e)))?
    .map_err(ApiError::from)?
    .ok_or_else(|| ApiError::NotFound(format!("machine '{}' not found", name)))?;

    Ok(Json(machine_status_to_info(&status)))
}

/// Start a machine.
#[utoipa::path(
    post,
    path = "/api/v1/machines/{name}/start",
    tag = "Machines",
    params(
        ("name" = String, Path, description = "Machine name")
    ),
    responses(
        (status = 200, description = "Machine started", body = MachineInfo),
        (status = 404, description = "Machine not found", body = ApiErrorResponse),
        (status = 500, description = "Failed to start", body = ApiErrorResponse)
    )
)]
pub async fn start_machine(
    State(state): State<Arc<ApiState>>,
    Path(name): Path<String>,
    body: Option<Json<StartMachineRequest>>,
) -> Result<Json<MachineInfo>, ApiError> {
    let lifecycle = state.lifecycle_lock(&name);
    let _guard = lifecycle.lock().await;

    let db = state.db().clone();
    let start_request = body.map(|Json(body)| body).unwrap_or_default();
    let status = tokio::task::spawn_blocking({
        let name = name.clone();
        move || {
            let mut request = StartMachine::new(name);
            request.forkable = start_request.forkable;
            request.proxy = start_request.proxy;
            request.no_proxy = start_request.no_proxy;
            LocalMachineService::with_db(db).start(request)
        }
    })
    .await
    .map_err(|e| ApiError::internal(format!("task error: {}", e)))?
    .map_err(ApiError::from)?;

    let manager = tokio::task::spawn_blocking({
        let name = name.clone();
        let storage_gb = status.record.storage_gb;
        let overlay_gb = status.record.overlay_gb;
        move || AgentManager::for_vm_with_sizes(&name, storage_gb, overlay_gb)
    })
    .await
    .map_err(|e| ApiError::internal(format!("task error: {}", e)))?
    .map_err(|e| ApiError::internal(format!("failed to create agent manager: {}", e)))?;
    state.insert_machine(&name, machine_entry_from_record(&status.record, manager));

    Ok(Json(machine_status_to_info(&status)))
}

/// Fork a running forkable machine.
#[utoipa::path(
    post,
    path = "/api/v1/machines/{name}/fork",
    tag = "Machines",
    params(
        ("name" = String, Path, description = "Golden machine name")
    ),
    request_body = ForkMachineRequest,
    responses(
        (status = 200, description = "Machine forked", body = MachineInfo),
        (status = 400, description = "Invalid request", body = ApiErrorResponse),
        (status = 404, description = "Machine not found", body = ApiErrorResponse),
        (status = 409, description = "Clone already exists or golden not forkable", body = ApiErrorResponse)
    )
)]
pub async fn fork_machine(
    State(state): State<Arc<ApiState>>,
    Path(name): Path<String>,
    Json(req): Json<ForkMachineRequest>,
) -> Result<Json<MachineInfo>, ApiError> {
    let lifecycle = state.lifecycle_lock(&name);
    let _guard = lifecycle.lock().await;

    let mut request = ForkMachine::new(name.clone(), req.clone.clone());
    request.ports = req
        .ports
        .iter()
        .map(crate::agent::PortMapping::from)
        .collect();

    let db = state.db().clone();
    let status =
        tokio::task::spawn_blocking(move || LocalMachineService::with_db(db).fork(request))
            .await
            .map_err(|e| ApiError::internal(format!("task error: {}", e)))?
            .map_err(ApiError::from)?;

    let manager = tokio::task::spawn_blocking({
        let name = status.name.clone();
        let storage_gb = status.record.storage_gb;
        let overlay_gb = status.record.overlay_gb;
        move || AgentManager::for_vm_with_sizes(&name, storage_gb, overlay_gb)
    })
    .await
    .map_err(|e| ApiError::internal(format!("task error: {}", e)))?
    .map_err(|e| ApiError::internal(format!("failed to create agent manager: {}", e)))?;
    state.insert_machine(
        &status.name,
        machine_entry_from_record(&status.record, manager),
    );

    Ok(Json(machine_status_to_info(&status)))
}

/// Stop a machine.
#[utoipa::path(
    post,
    path = "/api/v1/machines/{name}/stop",
    tag = "Machines",
    params(
        ("name" = String, Path, description = "Machine name")
    ),
    responses(
        (status = 200, description = "Machine stopped", body = MachineInfo),
        (status = 404, description = "Machine not found", body = ApiErrorResponse),
        (status = 500, description = "Failed to stop", body = ApiErrorResponse)
    )
)]
pub async fn stop_machine(
    State(state): State<Arc<ApiState>>,
    Path(name): Path<String>,
) -> Result<Json<MachineInfo>, ApiError> {
    let lifecycle = state.lifecycle_lock(&name);
    let _guard = lifecycle.lock().await;

    let db = state.db().clone();
    let status = tokio::task::spawn_blocking({
        let name = name.clone();
        move || LocalMachineService::with_db(db).stop(StopMachine::new(name))
    })
    .await
    .map_err(|e| ApiError::internal(format!("task error: {}", e)))?
    .map_err(ApiError::from)?;

    if let Ok(entry) = state.get_machine(&name) {
        entry.lock().manager.mark_stopped();
    }

    Ok(Json(machine_status_to_info(&status)))
}

/// Gracefully stop every running VM before the server exits. Opt-in via
/// `SMOLVM_DRAIN_ON_SHUTDOWN` (set by cloud workers); off by default so a dev
/// Ctrl-C or `serve` restart leaves VMs running for reconnect. Without draining,
/// a host teardown (e.g. autoscaler scale-in) hard-kills running VMs; draining
/// stops them cleanly — flushing disk state and marking them stopped so the
/// control plane can reschedule. Best-effort, concurrent, and bounded so it fits
/// inside the host's termination grace period.
pub async fn drain_machines(state: &Arc<ApiState>) {
    let running: Vec<(String, Option<i32>, Option<u64>)> = match state.db().list_vms() {
        Ok(vms) => vms
            .into_iter()
            .filter(|(_, r)| r.actual_state() == RecordState::Running && r.is_process_alive())
            .map(|(name, r)| (name, r.pid, r.pid_start_time))
            .collect(),
        Err(e) => {
            tracing::error!(error = %e, "drain: failed to list machines");
            return;
        }
    };
    if running.is_empty() {
        return;
    }
    tracing::info!(
        count = running.len(),
        "draining running machines before shutdown"
    );

    let mut handles = Vec::with_capacity(running.len());
    for (name, pid, pid_start_time) in running {
        let state = state.clone();
        handles.push(tokio::spawn(async move {
            let name_for_kill = name.clone();
            let entry = state.get_machine(&name).ok();
            let stopped = tokio::task::spawn_blocking(move || {
                // Prefer the registered manager (holds the flock); fall back to a
                // PID-verified signal — same path as the stop handler.
                let via_manager = entry
                    .as_ref()
                    .map(|e| e.lock().manager.stop().is_ok())
                    .unwrap_or(false);
                via_manager || shutdown_machine_process(&name_for_kill, pid, pid_start_time)
            })
            .await
            .unwrap_or(false);
            if let Ok(entry) = state.get_machine(&name) {
                entry.lock().manager.mark_stopped();
            }
            let _ = state.db().update_vm(&name, |r| {
                r.state = RecordState::Stopped;
                r.pid = None;
                r.pid_start_time = None;
            });
            tracing::info!(machine = %name, stopped, "drain: machine stopped");
        }));
    }

    let drain_all = async {
        for h in handles {
            let _ = h.await;
        }
    };
    if tokio::time::timeout(std::time::Duration::from_secs(25), drain_all)
        .await
        .is_err()
    {
        tracing::warn!("drain: deadline reached before all machines stopped");
    }
}

/// Delete a machine.
#[utoipa::path(
    delete,
    path = "/api/v1/machines/{name}",
    tag = "Machines",
    params(
        ("name" = String, Path, description = "Machine name")
    ),
    responses(
        (status = 200, description = "Machine deleted", body = DeleteResponse),
        (status = 404, description = "Machine not found", body = ApiErrorResponse),
        (status = 500, description = "Failed to delete", body = ApiErrorResponse)
    )
)]
pub async fn delete_machine(
    State(state): State<Arc<ApiState>>,
    Path(name): Path<String>,
    Query(query): Query<DeleteQuery>,
) -> Result<Json<DeleteResponse>, ApiError> {
    let lifecycle = state.lifecycle_lock(&name);
    let _guard = lifecycle.lock().await;

    let db = state.db().clone();
    tokio::task::spawn_blocking({
        let name = name.clone();
        move || {
            let mut request = DeleteMachine::new(name);
            request.break_dependent_clones = query.force;
            LocalMachineService::with_db(db).delete(request)
        }
    })
    .await
    .map_err(|e| ApiError::internal(format!("task error: {}", e)))?
    .map_err(ApiError::from)?;

    let _ = state.forget_machine(&name);
    Ok(Json(DeleteResponse { deleted: name }))
}

/// Update a stopped machine.
#[utoipa::path(
    patch,
    path = "/api/v1/machines/{name}",
    tag = "Machines",
    params(
        ("name" = String, Path, description = "Machine name")
    ),
    request_body = UpdateMachineRequest,
    responses(
        (status = 200, description = "Machine updated", body = MachineInfo),
        (status = 400, description = "Invalid request", body = ApiErrorResponse),
        (status = 404, description = "Machine not found", body = ApiErrorResponse),
        (status = 409, description = "Machine must be stopped", body = ApiErrorResponse)
    )
)]
pub async fn update_machine(
    State(state): State<Arc<ApiState>>,
    Path(name): Path<String>,
    Json(req): Json<UpdateMachineRequest>,
) -> Result<Json<MachineInfo>, ApiError> {
    let mut update = UpdateMachine::new(name.clone());
    update.add_mounts = req
        .add_mounts
        .iter()
        .map(HostMount::try_from)
        .collect::<crate::Result<Vec<_>>>()
        .map_err(|e| ApiError::BadRequest(e.to_string()))?;
    update.remove_mounts = req
        .remove_mounts
        .iter()
        .map(HostMount::try_from)
        .collect::<crate::Result<Vec<_>>>()
        .map_err(|e| ApiError::BadRequest(e.to_string()))?;
    update.add_ports = req
        .add_ports
        .iter()
        .map(crate::agent::PortMapping::from)
        .collect();
    update.remove_ports = req
        .remove_ports
        .iter()
        .map(crate::agent::PortMapping::from)
        .collect();
    update.cpus = req.cpus;
    update.memory_mib = req.mem;
    match req.network {
        Some(true) => update.enable_network = true,
        Some(false) => update.disable_network = true,
        None => {}
    }
    match req.gpu {
        Some(true) => update.enable_gpu = true,
        Some(false) => update.disable_gpu = true,
        None => {}
    }
    update.storage_gb = req.storage_gb;
    update.overlay_gb = req.overlay_gb;
    update.set_env = EnvVar::to_tuples(&req.env);
    update.remove_env = req.remove_env;
    update.workdir = req.workdir;
    update.allowed_cidrs = req.allowed_cidrs;
    update.dns_filter_hosts = req.dns_filter_hosts;
    match req.ssh_agent {
        Some(true) => update.enable_ssh_agent = true,
        Some(false) => update.disable_ssh_agent = true,
        None => {}
    }

    let db = state.db().clone();
    let status = tokio::task::spawn_blocking(move || {
        LocalMachineService::with_db(db)
            .update(update)
            .map(|result| result.status)
    })
    .await
    .map_err(|e| ApiError::internal(format!("task error: {}", e)))?
    .map_err(ApiError::from)?;

    Ok(Json(machine_status_to_info(&status)))
}

/// Resize a machine's disk resources.
#[utoipa::path(
    post,
    path = "/api/v1/machines/{name}/resize",
    tag = "Machines",
    params(
        ("name" = String, Path, description = "Machine name")
    ),
    request_body = ResizeMachineRequest,
    responses(
        (status = 200, description = "Machine resized", body = MachineInfo),
        (status = 400, description = "Invalid request", body = ApiErrorResponse),
        (status = 404, description = "Machine not found", body = ApiErrorResponse),
        (status = 409, description = "Machine is running", body = ApiErrorResponse),
        (status = 500, description = "Resize failed", body = ApiErrorResponse)
    )
)]
pub async fn resize_machine(
    State(state): State<Arc<ApiState>>,
    Path(name): Path<String>,
    Json(req): Json<ResizeMachineRequest>,
) -> Result<Json<MachineInfo>, ApiError> {
    if req.storage_gb.is_none() && req.overlay_gb.is_none() {
        return Err(ApiError::BadRequest(
            "at least one of storageGb or overlayGb must be specified".into(),
        ));
    }

    let db = state.db().clone();
    let status = tokio::task::spawn_blocking({
        let name = name.clone();
        move || {
            let mut update = UpdateMachine::new(name);
            update.storage_gb = req.storage_gb;
            update.overlay_gb = req.overlay_gb;
            LocalMachineService::with_db(db)
                .update(update)
                .map(|result| result.status)
        }
    })
    .await
    .map_err(|e| ApiError::internal(format!("task error: {}", e)))?
    .map_err(ApiError::from)?;

    Ok(Json(machine_status_to_info(&status)))
}

async fn pull_from_registry(
    registry_ref: &str,
    identity_token: Option<&str>,
) -> Result<String, ApiError> {
    let parsed = crate::registry::Reference::parse(registry_ref)
        .map_err(|e| ApiError::BadRequest(format!("invalid registry reference: {}", e)))?;

    let settings = crate::settings::SmolSettings::load()
        .map_err(|e| ApiError::internal(format!("load settings: {}", e)))?;

    let effective_registry = settings
        .machines
        .get_mirror(&parsed.registry)
        .unwrap_or(&parsed.registry);
    let api_host = match effective_registry {
        "docker.io" => "registry-1.docker.io",
        h => h,
    };
    let base_url = if smolvm_registry::is_local_registry(api_host) {
        format!("http://{}", api_host)
    } else {
        format!("https://{}", api_host)
    };

    let mut client = smolvm_registry::RegistryClient::new(base_url);

    // A request-supplied identity token (the control plane's short-lived,
    // tenant-scoped pull token) takes precedence over any persisted credential.
    if let Some(token) = identity_token {
        client = client.with_identity_token(token.to_string());
    } else if let Some(entry) = settings.machines.registries.get(effective_registry) {
        if let Some(ref token) = entry.identity_token {
            client = client.with_identity_token(token.clone());
        }
    }

    let cache = smolvm_registry::BlobCache::open_default()
        .map_err(|e| ApiError::internal(format!("blob cache: {}", e)))?;

    let repo = parsed.repository();
    let tag_or_digest = registry_reference_tag_or_digest(&parsed);

    tracing::info!(
        registry_ref = %registry_ref,
        repo = %repo,
        reference = %tag_or_digest,
        "pulling .smolmachine from registry"
    );

    let result = smolvm_registry::pull(&client, &repo, tag_or_digest, None, &cache)
        .await
        .map_err(|e| ApiError::internal(format!("registry pull failed: {}", e)))?;

    tracing::info!(path = %result.path.display(), cached = result.cached, "pull complete");

    Ok(result.path.to_string_lossy().into_owned())
}

fn registry_reference_tag_or_digest(parsed: &crate::registry::Reference) -> &str {
    parsed
        .digest
        .as_deref()
        .or(parsed.tag.as_deref())
        .unwrap_or("latest")
}

fn resolve_create_resources(
    req: &CreateMachineRequest,
    manifest_cpus: u8,
    manifest_mem: u32,
) -> (u8, u32) {
    (
        req.cpus.unwrap_or(manifest_cpus),
        req.mem.unwrap_or(manifest_mem),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::SmolvmDb;
    use tempfile::TempDir;

    fn info_from_record(name: &str, record: VmRecord, state: RecordState) -> MachineInfo {
        machine_status_to_info(&MachineStatus {
            name: name.to_string(),
            state,
            record,
        })
    }

    #[test]
    fn test_machine_status_to_info() {
        let record = VmRecord::new(
            "test-vm".to_string(),
            2,
            1024,
            vec![
                ("/host/path".to_string(), "/guest/path".to_string(), false),
                ("/host/ro".to_string(), "/guest/ro".to_string(), true),
            ],
            vec![(8080, 80), (3000, 3000)],
            false,
        );

        let info = info_from_record("test-vm", record, RecordState::Created);

        assert_eq!(info.name, "test-vm");
        assert_eq!(info.state, "created");
        assert_eq!(info.cpus, 2);
        assert_eq!(info.mem, 1024);
        assert_eq!(info.mounts.len(), 2);
        assert_eq!(info.ports.len(), 2);
        assert!(!info.network);
        assert!(info.pid.is_none());
    }

    #[test]
    fn test_machine_status_to_info_with_running_state() {
        let mut record = VmRecord::new("running-vm".to_string(), 1, 512, vec![], vec![], false);
        record.state = RecordState::Running;
        record.pid = Some(12345);

        let info = info_from_record("running-vm", record, RecordState::Running);

        assert_eq!(info.name, "running-vm");
        // Note: actual_state() checks if process is alive, which won't be true in test
        // So it will show as "stopped" even though record state is Running
        assert_eq!(info.cpus, 1);
        assert_eq!(info.mem, 512);
        assert_eq!(info.mounts.len(), 0);
        assert_eq!(info.ports.len(), 0);
    }

    #[test]
    fn test_machine_status_to_info_default_values() {
        let record = VmRecord::new("minimal-vm".to_string(), 1, 512, vec![], vec![], false);

        let info = info_from_record("minimal-vm", record, RecordState::Created);

        assert_eq!(info.name, "minimal-vm");
        assert_eq!(info.state, "created");
        assert_eq!(info.cpus, 1);
        assert_eq!(info.mem, 512);
        assert_eq!(info.mounts.len(), 0);
        assert_eq!(info.ports.len(), 0);
        assert!(!info.network);
        assert!(info.pid.is_none());
        assert!(info.created_at > 0);
    }

    #[test]
    fn test_machine_status_to_info_with_network() {
        let record = VmRecord::new("network-vm".to_string(), 1, 512, vec![], vec![], true);

        let info = info_from_record("network-vm", record, RecordState::Created);

        assert_eq!(info.name, "network-vm");
        assert!(info.network);
    }

    #[test]
    fn registry_reference_uses_digest_before_tag_or_latest() {
        let digest = "sha256:abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

        let digest_ref =
            crate::registry::Reference::parse(&format!("python-dev@{digest}")).unwrap();
        assert_eq!(registry_reference_tag_or_digest(&digest_ref), digest);

        let tagged_ref = crate::registry::Reference::parse("python-dev:v1").unwrap();
        assert_eq!(registry_reference_tag_or_digest(&tagged_ref), "v1");

        let latest_ref = crate::registry::Reference::parse("python-dev").unwrap();
        assert_eq!(registry_reference_tag_or_digest(&latest_ref), "latest");
    }

    fn minimal_create_request() -> CreateMachineRequest {
        CreateMachineRequest {
            name: Some("test-vm".to_string()),
            cpus: None,
            mem: None,
            mounts: vec![],
            ports: vec![],
            network: false,
            gpu: false,
            storage_gb: None,
            overlay_gb: None,
            allowed_cidrs: None,
            restart: None,
            image: None,
            from: None,
            registry_ref: None,
            registry_identity_token: None,
            secrets: Default::default(),
        }
    }

    #[test]
    fn create_resources_use_high_defaults_when_omitted() {
        let req = minimal_create_request();

        assert_eq!(
            resolve_create_resources(
                &req,
                crate::data::resources::DEFAULT_MICROVM_CPU_COUNT,
                crate::data::resources::DEFAULT_MICROVM_MEMORY_MIB,
            ),
            (
                crate::data::resources::DEFAULT_MICROVM_CPU_COUNT,
                crate::data::resources::DEFAULT_MICROVM_MEMORY_MIB,
            )
        );
    }

    #[test]
    fn create_resources_preserve_manifest_defaults_when_omitted() {
        let req = minimal_create_request();

        assert_eq!(resolve_create_resources(&req, 6, 12_288), (6, 12_288));
    }

    #[test]
    fn create_resources_explicit_api_values_override_manifest_defaults() {
        let mut req = minimal_create_request();
        req.cpus = Some(2);
        req.mem = Some(2048);

        assert_eq!(resolve_create_resources(&req, 6, 12_288), (2, 2048));
    }

    #[test]
    fn create_request_deserialization_keeps_resource_omission_distinct() {
        let req: CreateMachineRequest = serde_json::from_value(serde_json::json!({
            "name": "api-vm"
        }))
        .unwrap();

        assert_eq!(req.cpus, None);
        assert_eq!(req.mem, None);

        let req: CreateMachineRequest = serde_json::from_value(serde_json::json!({
            "name": "api-vm",
            "cpus": 2,
            "memoryMb": 2048
        }))
        .unwrap();

        assert_eq!(req.cpus, Some(2));
        assert_eq!(req.mem, Some(2048));
    }

    /// Helper to create a test database and API state.
    fn setup_test_state() -> (TempDir, Arc<ApiState>) {
        let dir = TempDir::new().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let db = SmolvmDb::open_at(&db_path).expect("failed to open test db");
        let state = Arc::new(ApiState::with_db(db));
        (dir, state)
    }

    #[tokio::test]
    async fn test_resize_validation_shrink_storage_rejected() {
        let (_dir, state) = setup_test_state();
        let db = state.db();
        create_test_vm(db, "test-vm", Some(20), Some(5));

        let req = ResizeMachineRequest {
            storage_gb: Some(10),
            overlay_gb: None,
        };
        let result = resize_machine(State(state), Path("test-vm".to_string()), Json(req)).await;
        assert!(matches!(result.unwrap_err(), ApiError::BadRequest(_)));
    }

    #[tokio::test]
    async fn test_resize_validation_no_params_rejected() {
        let (_dir, state) = setup_test_state();
        let db = state.db();
        create_test_vm(db, "test-vm", Some(20), Some(5));

        let req = ResizeMachineRequest {
            storage_gb: None,
            overlay_gb: None,
        };
        let result = resize_machine(State(state), Path("test-vm".to_string()), Json(req)).await;
        assert!(matches!(result.unwrap_err(), ApiError::BadRequest(_)));
    }

    #[tokio::test]
    async fn test_resize_not_found() {
        let (_dir, state) = setup_test_state();
        let req = ResizeMachineRequest {
            storage_gb: Some(30),
            overlay_gb: None,
        };
        let result = resize_machine(State(state), Path("nonexistent".to_string()), Json(req)).await;
        assert!(matches!(result.unwrap_err(), ApiError::NotFound(_)));
    }

    /// Helper to create a VM record in the database.
    fn create_test_vm(db: &SmolvmDb, name: &str, storage_gb: Option<u64>, overlay_gb: Option<u64>) {
        let mut record = VmRecord::new(name.to_string(), 1, 512, vec![], vec![], false);
        record.storage_gb = storage_gb;
        record.overlay_gb = overlay_gb;
        db.insert_vm(name, &record)
            .expect("failed to insert test vm");
    }
}

/// Stream monitor events for a machine as Server-Sent Events.
#[utoipa::path(
    get,
    path = "/api/v1/machines/{name}/monitor",
    tag = "Machines",
    params(
        ("name" = String, Path, description = "Machine name"),
        MonitorQuery
    ),
    responses(
        (status = 200, description = "Monitor event stream", content_type = "text/event-stream"),
        (status = 404, description = "Machine not found", body = ApiErrorResponse)
    )
)]
pub async fn monitor_machine(
    State(state): State<Arc<ApiState>>,
    Path(name): Path<String>,
    Query(query): Query<MonitorQuery>,
) -> Result<Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let mut request = MonitorMachine::new(name.clone());
    request.restart_policy = query
        .restart
        .as_deref()
        .map(str::parse)
        .transpose()
        .map_err(|e: String| ApiError::BadRequest(e))?;
    request.health_cmd = query
        .health_cmd
        .map(|command| vec!["sh".to_string(), "-c".to_string(), command]);
    if let Some(timeout) = query.health_timeout_secs {
        request.health_timeout = std::time::Duration::from_secs(timeout);
    }
    if let Some(interval) = query.interval_secs {
        request.interval = std::time::Duration::from_secs(interval);
    }
    if let Some(retries) = query.health_retries {
        request.health_retries = retries;
    }

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_for_thread = stop.clone();
    let db = state.db().clone();
    tokio::task::spawn_blocking(move || {
        let service = LocalMachineService::with_db(db);
        let mut on_event = |event| {
            let _ = tx.send(monitor_event_to_sse(event));
        };
        if let Err(error) = service.monitor(request, &mut on_event, &|| {
            stop_for_thread.load(Ordering::SeqCst)
        }) {
            let _ = tx.send(
                Event::default()
                    .event("error")
                    .data(serde_json::json!({ "message": error.to_string() }).to_string()),
            );
        }
    });

    struct StopOnDrop(Arc<AtomicBool>);
    impl Drop for StopOnDrop {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    let guard = StopOnDrop(stop);
    let stream = async_stream::stream! {
        let _guard = guard;
        while let Some(event) = rx.recv().await {
            yield Ok(event);
        }
    };
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

fn monitor_event_to_sse(event: MonitorEvent) -> Event {
    match event {
        MonitorEvent::Starting { name } => Event::default()
            .event("starting")
            .data(serde_json::json!({ "name": name }).to_string()),
        MonitorEvent::Monitoring {
            name,
            policy,
            interval_secs,
            health,
        } => Event::default().event("monitoring").data(
            serde_json::json!({
                "name": name,
                "policy": policy.to_string(),
                "intervalSecs": interval_secs,
                "health": health.map(|h| serde_json::json!({
                    "command": h.command,
                    "timeoutSecs": h.timeout_secs,
                    "retries": h.retries,
                })),
            })
            .to_string(),
        ),
        MonitorEvent::SuspendDetected { sleep_secs } => Event::default()
            .event("suspend")
            .data(serde_json::json!({ "sleepSecs": sleep_secs }).to_string()),
        MonitorEvent::HealthRecovered => Event::default()
            .event("healthRecovered")
            .data(serde_json::json!({}).to_string()),
        MonitorEvent::HealthFailed {
            exit_code,
            consecutive,
            retries,
            stderr,
        } => Event::default().event("healthFailed").data(
            serde_json::json!({
                "exitCode": exit_code,
                "consecutive": consecutive,
                "retries": retries,
                "stderr": stderr,
            })
            .to_string(),
        ),
        MonitorEvent::HealthError {
            consecutive,
            retries,
            error,
        } => Event::default().event("healthError").data(
            serde_json::json!({
                "consecutive": consecutive,
                "retries": retries,
                "error": error,
            })
            .to_string(),
        ),
        MonitorEvent::AgentUnreachable {
            consecutive,
            retries,
        } => Event::default().event("agentUnreachable").data(
            serde_json::json!({
                "consecutive": consecutive,
                "retries": retries,
            })
            .to_string(),
        ),
        MonitorEvent::UnhealthyStopping => Event::default()
            .event("unhealthyStopping")
            .data(serde_json::json!({}).to_string()),
        MonitorEvent::MachineExited { exit_code } => Event::default()
            .event("machineExited")
            .data(serde_json::json!({ "exitCode": exit_code }).to_string()),
        MonitorEvent::Restarting {
            attempt,
            backoff_secs,
        } => Event::default().event("restarting").data(
            serde_json::json!({ "attempt": attempt, "backoffSecs": backoff_secs }).to_string(),
        ),
        MonitorEvent::Restarted => Event::default()
            .event("restarted")
            .data(serde_json::json!({}).to_string()),
        MonitorEvent::RestartFailed { error } => Event::default()
            .event("restartFailed")
            .data(serde_json::json!({ "error": error }).to_string()),
        MonitorEvent::NotRestarting {
            policy,
            count,
            max_retries,
        } => Event::default().event("notRestarting").data(
            serde_json::json!({
                "policy": policy.to_string(),
                "count": count,
                "maxRetries": max_retries,
            })
            .to_string(),
        ),
        MonitorEvent::Stopped { name } => Event::default()
            .event("stopped")
            .data(serde_json::json!({ "name": name }).to_string()),
    }
}

/// Run a network connectivity diagnostic inside a machine.
#[utoipa::path(
    post,
    path = "/api/v1/machines/{name}/network-test",
    tag = "Machines",
    params(
        ("name" = String, Path, description = "Machine name")
    ),
    request_body = NetworkTestRequest,
    responses(
        (status = 200, description = "Network diagnostic result", body = NetworkTestResponse),
        (status = 400, description = "Invalid request", body = ApiErrorResponse),
        (status = 404, description = "Machine not found", body = ApiErrorResponse)
    )
)]
pub async fn network_test(
    State(state): State<Arc<ApiState>>,
    Path(name): Path<String>,
    trace_id: Option<axum::Extension<crate::api::TraceId>>,
    Json(req): Json<NetworkTestRequest>,
) -> Result<Json<NetworkTestResponse>, ApiError> {
    if req.url.is_empty() {
        return Err(ApiError::BadRequest("url cannot be empty".into()));
    }
    let mut request = NetworkTestMachine::new(name, req.url);
    request.start_if_needed = req.start_if_needed;
    request.trace_id = trace_id.map(|t| t.0 .0.clone());
    let db = state.db().clone();
    let result =
        tokio::task::spawn_blocking(move || LocalMachineService::with_db(db).network_test(request))
            .await
            .map_err(|e| ApiError::internal(format!("task error: {}", e)))?
            .map_err(ApiError::from)?;
    Ok(Json(NetworkTestResponse { result }))
}

/// Return the host data directory path for a machine.
#[utoipa::path(
    get,
    path = "/api/v1/machines/{name}/data-dir",
    tag = "Machines",
    params(
        ("name" = String, Path, description = "Machine name")
    ),
    responses(
        (status = 200, description = "Machine data directory", body = DataDirResponse),
        (status = 404, description = "Machine not found", body = ApiErrorResponse)
    )
)]
pub async fn data_dir(
    State(state): State<Arc<ApiState>>,
    Path(name): Path<String>,
) -> Result<Json<DataDirResponse>, ApiError> {
    let request = DataDirMachine::new(name.clone());
    let db = state.db().clone();
    let path =
        tokio::task::spawn_blocking(move || LocalMachineService::with_db(db).data_dir(request))
            .await
            .map_err(|e| ApiError::internal(format!("task error: {}", e)))?
            .map_err(ApiError::from)?;
    Ok(Json(DataDirResponse {
        name,
        path: path.to_string_lossy().into_owned(),
    }))
}
