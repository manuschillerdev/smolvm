//! Shared helpers for machine CLI commands.
//!
//! The `machine` subcommand exposes lifecycle commands
//! (create, start, stop, delete, ls). This module provides the common
//! implementations used by those commands.

use crate::cli::truncate;
use smolvm::agent::AgentManager;
use smolvm::config::RecordState;
use smolvm::data::network::PortMapping;
use smolvm::data::storage::HostMount;
use smolvm::db::SmolvmDb;
use smolvm::machine::{
    CreateMachine, GetMachine, ListMachines, LocalMachineService, MachineService, MachineStatus,
};
use smolvm::network::NetworkBackend;
use smolvm::secrets::SecretRef;
use smolvm::storage::{DEFAULT_OVERLAY_SIZE_GIB, DEFAULT_STORAGE_SIZE_GIB};
use std::collections::BTreeMap;

// ============================================================================
// Shared helpers
// ============================================================================

/// Resolve an optional VM name: if no name is given and a VM named "default"
/// exists in the config database, return `Some("default")` so callers route
/// through the named-VM code path (which loads config, init commands, network
/// settings, etc.). Otherwise returns the input unchanged.
pub fn resolve_vm_name(name: Option<String>) -> smolvm::Result<Option<String>> {
    if name.is_some() {
        return Ok(name);
    }
    // Use direct DB lookup instead of SmolvmConfig::load() to avoid
    // loading all config + all VMs just to check if "default" exists.
    let db = SmolvmDb::open()?;
    if db.get_vm("default")?.is_some() {
        Ok(Some("default".to_string()))
    } else {
        Ok(None)
    }
}

/// Get the agent manager for an optional name (default if `None`).
///
/// When no name is given, uses `AgentManager::new_default()` which is
/// canonicalized to `for_vm("default")` — same socket/PID/storage paths
/// regardless of whether the caller specifies a name or not.
pub fn get_vm_manager(name: &Option<String>) -> smolvm::Result<AgentManager> {
    if let Some(name) = name {
        AgentManager::for_vm(name)
    } else {
        AgentManager::new_default()
    }
}

/// Return the display label for an optional VM name.
pub fn vm_label(name: &Option<String>) -> String {
    name.as_deref().unwrap_or("default").to_string()
}

// ============================================================================
// Create
// ============================================================================

/// Parameters for [`create_vm`].
pub struct CreateVmParams {
    pub name: String,
    pub image: Option<String>,
    pub entrypoint: Vec<String>,
    pub cmd: Vec<String>,
    pub cpus: u8,
    pub mem: u32,
    pub volume: Vec<String>,
    pub port: Vec<PortMapping>,
    pub net: bool,
    pub network_backend: Option<NetworkBackend>,
    pub init: Vec<String>,
    pub env: Vec<String>,
    pub workdir: Option<String>,
    pub storage_gb: Option<u64>,
    pub overlay_gb: Option<u64>,
    pub allowed_cidrs: Option<Vec<String>>,
    pub restart_policy: Option<smolvm::config::RestartPolicy>,
    pub restart_max_retries: Option<u32>,
    pub restart_max_backoff_secs: Option<u64>,
    pub health_cmd: Option<Vec<String>>,
    pub health_interval_secs: Option<u64>,
    pub health_timeout_secs: Option<u64>,
    pub health_retries: Option<u32>,
    pub health_startup_grace_secs: Option<u64>,
    pub ssh_agent: bool,
    /// Enable GPU acceleration (virtio-gpu with Venus/Vulkan).
    pub gpu: bool,
    /// GPU VRAM size in MiB (None = default). Ignored when gpu is false.
    pub gpu_vram_mib: Option<u32>,
    /// Hostnames for DNS filtering (from --allow-host / [network].allow_hosts).
    pub dns_filter_hosts: Option<Vec<String>>,
    /// Absolute path to .smolmachine sidecar (for machines created with --from).
    pub source_smolmachine: Option<String>,
    /// Secret refs from Smolfile `[secrets]`. The refs themselves are
    /// persisted to the VM record (they are not sensitive); resolved
    /// plaintext values are produced per-launch and never touch the DB.
    pub secret_refs: BTreeMap<String, SecretRef>,
}

/// Create a named machine configuration (does not start it).
pub fn create_vm(params: CreateVmParams) -> smolvm::Result<()> {
    let request = create_machine_request(&params)?;
    LocalMachineService::new()?.create(request)?;
    print_create_success(&params);
    Ok(())
}

fn create_machine_request(params: &CreateVmParams) -> smolvm::Result<CreateMachine> {
    let mut request = CreateMachine::new(params.name.clone());
    request.image = params.image.clone();
    request.entrypoint = params.entrypoint.clone();
    request.cmd = params.cmd.clone();
    request.cpus = params.cpus;
    request.memory_mib = params.mem;
    request.mounts = HostMount::parse(&params.volume)?;
    request.ports = params.port.clone();
    request.net = params.net;
    request.network_backend = params.network_backend;
    request.init = params.init.clone();
    request.env = params.env.clone();
    request.workdir = params.workdir.clone();
    request.storage_gb = params.storage_gb;
    request.overlay_gb = params.overlay_gb;
    request.allowed_cidrs = params.allowed_cidrs.clone();
    request.restart_policy = params.restart_policy.clone();
    request.restart_max_retries = params.restart_max_retries;
    request.restart_max_backoff_secs = params.restart_max_backoff_secs;
    request.health_cmd = params.health_cmd.clone();
    request.health_interval_secs = params.health_interval_secs;
    request.health_timeout_secs = params.health_timeout_secs;
    request.health_retries = params.health_retries;
    request.health_startup_grace_secs = params.health_startup_grace_secs;
    request.ssh_agent = params.ssh_agent;
    request.gpu = params.gpu;
    request.gpu_vram_mib = params.gpu_vram_mib;
    request.dns_filter_hosts = params.dns_filter_hosts.clone();
    request.source_smolmachine = params.source_smolmachine.clone();
    request.secret_refs = params.secret_refs.clone();
    Ok(request)
}

pub(crate) fn print_create_success(params: &CreateVmParams) {
    println!("Created machine: {}", params.name);
    println!("  CPUs: {}, Memory: {} MiB", params.cpus, params.mem);
    if !params.volume.is_empty() {
        println!("  Mounts: {}", params.volume.len());
    }
    if !params.port.is_empty() {
        println!("  Ports: {}", params.port.len());
    }
    if !params.init.is_empty() {
        println!("  Init commands: {}", params.init.len());
    }
    println!(
        "\nUse 'smolvm machine start --name {}' to start the machine",
        params.name
    );
    println!(
        "Then use 'smolvm machine exec --name {} -- <command>' to run commands",
        params.name
    );
}

// ============================================================================
// Status
// ============================================================================

/// Show status of a named or default machine.
///
/// The `extra` callback is invoked when the VM is running, allowing callers
/// to display additional information (e.g., machine lists containers).
pub fn status_vm<F>(name: &Option<String>, extra: F) -> smolvm::Result<()>
where
    F: FnOnce(&AgentManager),
{
    let label = vm_label(name);
    let Some(status) = LocalMachineService::new()?.status(GetMachine::new(label.clone()))? else {
        if name.is_some() {
            return Err(smolvm::Error::vm_not_found(&label));
        }
        println!("Machine '{}': not running", label);
        return Ok(());
    };

    if status.state == RecordState::Running {
        let manager = get_vm_manager(&Some(label.clone()))?;
        let pid_suffix = crate::cli::format_pid_suffix(status.record.pid);
        println!("Machine '{}': running{}", label, pid_suffix);
        extra(&manager);
        manager.detach();
    } else {
        println!("Machine '{}': {}", label, status.state);
    }

    Ok(())
}

/// Build the per-machine JSON object shared by `machine list --json` and
/// `machine status --json` so the two outputs never drift apart.
fn machine_status_json(status: &MachineStatus) -> serde_json::Value {
    let name = &status.name;
    let record = &status.record;
    let actual_state = &status.state;
    // Expose the persisted health command as a single shell-friendly string
    // when it was stored as `["sh", "-c", "<cmd>"]`; otherwise a space-joined
    // argv so the field is always a string.
    let health_cmd_str = record.health_cmd.as_ref().map(|argv| {
        if argv.len() == 3 && argv[0] == "sh" && argv[1] == "-c" {
            argv[2].clone()
        } else {
            argv.join(" ")
        }
    });

    let mut obj = serde_json::json!({
        "name": name,
        "state": actual_state.to_string(),
        "cpus": record.cpus,
        "memory_mib": record.mem,
        "pid": record.pid,
        "mounts": record.mounts.len(),
        "ports": record.ports.len(),
        "created_at": record.created_at,
        "storage_gb": record.storage_gb,
        "overlay_gb": record.overlay_gb,
        "image": record.image,
        "entrypoint": record.entrypoint,
        "cmd": record.cmd,
        "ephemeral": record.ephemeral,
        "gpu": record.gpu.unwrap_or(false),
        "gpu_vram_mib": record.gpu_vram_mib,
        "restart_policy": record.restart.policy.to_string(),
        "restart_max_retries": record.restart.max_retries,
        "restart_count": record.restart.restart_count,
        "health_cmd": health_cmd_str,
        "health_interval_secs": record.health_interval_secs,
        "health_timeout_secs": record.health_timeout_secs,
        "health_retries": record.health_retries,
        "health_startup_grace_secs": record.health_startup_grace_secs,
    });
    obj.as_object_mut()
        .unwrap()
        .insert("network".into(), serde_json::json!(record.network));
    obj
}

/// Emit a single machine's status as JSON — the same object shape as
/// `machine list --json`. Errors if the machine does not exist.
pub fn status_vm_json(name: &Option<String>) -> smolvm::Result<()> {
    let label = vm_label(name);
    let status = LocalMachineService::new()?
        .status(GetMachine::new(label.clone()))?
        .ok_or_else(|| {
            smolvm::Error::config("machine status", format!("machine '{}' not found", label))
        })?;
    let json = serde_json::to_string_pretty(&machine_status_json(&status))
        .map_err(|e| smolvm::Error::config("serialize json", e.to_string()))?;
    println!("{}", json);
    Ok(())
}

// ============================================================================
// List
// ============================================================================

/// List all machines.
pub fn list_vms(verbose: bool, json: bool) -> smolvm::Result<()> {
    let statuses = LocalMachineService::new()?.list(ListMachines::default())?;

    let empty_label = "No machines found";

    if statuses.is_empty() {
        if !json {
            println!("{}", empty_label);
        } else {
            println!("[]");
        }
        return Ok(());
    }

    if json {
        let json_vms: Vec<_> = statuses.iter().map(machine_status_json).collect();
        let json = serde_json::to_string_pretty(&json_vms)
            .map_err(|e| smolvm::Error::config("serialize json", e.to_string()))?;
        println!("{}", json);
    } else {
        println!(
            "{:<20} {:<12} {:>5} {:>10} {:>7} {:>7} {:>8} {:>8}",
            "NAME", "STATE", "CPUS", "MEMORY", "MOUNTS", "PORTS", "STORAGE", "OVERLAY"
        );
        println!("{}", "-".repeat(88));

        for status in statuses {
            let name = &status.name;
            let record = &status.record;
            let state_display = if record.ephemeral {
                format!("{} (eph)", status.state)
            } else {
                status.state.to_string()
            };
            let storage_gb = record.storage_gb.unwrap_or(DEFAULT_STORAGE_SIZE_GIB);
            let overlay_gb = record.overlay_gb.unwrap_or(DEFAULT_OVERLAY_SIZE_GIB);
            println!(
                "{:<20} {:<12} {:>5} {:>10} {:>7} {:>7} {:>8} {:>8}",
                truncate(name, 18),
                state_display,
                record.cpus,
                format!("{} MiB", record.mem),
                record.mounts.len(),
                record.ports.len(),
                format!("{} GiB", storage_gb),
                format!("{} GiB", overlay_gb),
            );

            if verbose {
                if let Some(pid) = record.pid {
                    println!("  PID: {}", pid);
                }
                for (host, guest, ro) in &record.mounts {
                    let ro_str = if *ro { " (ro)" } else { "" };
                    println!("  Mount: {} -> {}{}", host, guest, ro_str);
                }
                for (host, guest) in &record.ports {
                    println!("  Port: {} -> {}", host, guest);
                }
                if record.network {
                    println!("  Network: enabled");
                }
                if record.gpu.unwrap_or(false) {
                    match record.gpu_vram_mib {
                        Some(vram) => println!("  GPU: enabled ({} MiB VRAM)", vram),
                        None => println!("  GPU: enabled"),
                    }
                }
                for cmd in &record.init {
                    println!("  Init: {}", cmd);
                }
                for (k, v) in &record.env {
                    println!("  Env: {}={}", k, v);
                }
                if let Some(wd) = &record.workdir {
                    println!("  Workdir: {}", wd);
                }
                let created =
                    std::time::UNIX_EPOCH + std::time::Duration::from_secs(record.created_at);
                println!("  Created: {}", humantime::format_rfc3339_seconds(created));
                println!();
            }
        }
    }

    Ok(())
}

// ============================================================================
// Resize
// ============================================================================

/// Resize a microVM's disk resources.
///
/// The VM must be stopped before resizing. Only expansion is supported
/// (no shrinking to prevent data loss).
/// Expand physical disk files for a VM. Does NOT update the DB record —
/// the caller is responsible for persisting the new sizes.
///
/// Returns a list of human-readable change descriptions for display.
/// Validates no-shrink and performs the physical I/O.
pub fn expand_disks(
    name: &str,
    record: &smolvm::config::VmRecord,
    new_storage_gb: Option<u64>,
    new_overlay_gb: Option<u64>,
) -> smolvm::Result<Vec<String>> {
    use smolvm::data::disk::{Overlay, Storage};
    use smolvm::storage::{expand_disk, DEFAULT_OVERLAY_SIZE_GIB, DEFAULT_STORAGE_SIZE_GIB};

    let current_storage_gb = record.storage_gb.unwrap_or(DEFAULT_STORAGE_SIZE_GIB);
    let current_overlay_gb = record.overlay_gb.unwrap_or(DEFAULT_OVERLAY_SIZE_GIB);

    // Validate no shrinking
    if let Some(s) = new_storage_gb {
        if s < current_storage_gb {
            return Err(smolvm::Error::config(
                "resize",
                format!(
                    "storage disk cannot be shrunk from {} GiB to {} GiB. Only expanding is supported to prevent data loss.",
                    current_storage_gb, s
                ),
            ));
        }
    }
    if let Some(o) = new_overlay_gb {
        if o < current_overlay_gb {
            return Err(smolvm::Error::config(
                "resize",
                format!(
                    "overlay disk cannot be shrunk from {} GiB to {} GiB. Only expanding is supported to prevent data loss.",
                    current_overlay_gb, o
                ),
            ));
        }
    }

    let manager = AgentManager::for_vm(name)
        .map_err(|e| smolvm::Error::agent("get agent manager", e.to_string()))?;

    let mut changes = Vec::new();

    if let Some(storage_gb) = new_storage_gb {
        if storage_gb > current_storage_gb {
            let storage_path = manager.storage_path();
            expand_disk::<Storage>(storage_path, storage_gb)
                .map_err(|e| smolvm::Error::storage("expand storage disk", e.to_string()))?;
            changes.push(format!(
                "  storage: {} GiB → {} GiB",
                current_storage_gb, storage_gb
            ));
        }
    }

    if let Some(overlay_gb) = new_overlay_gb {
        if overlay_gb > current_overlay_gb {
            let overlay_path = manager.overlay_path();
            expand_disk::<Overlay>(overlay_path, overlay_gb)
                .map_err(|e| smolvm::Error::storage("expand overlay disk", e.to_string()))?;
            changes.push(format!(
                "  overlay: {} GiB → {} GiB",
                current_overlay_gb, overlay_gb
            ));
        }
    }

    Ok(changes)
}

/// Legacy wrapper: expand disks AND update the DB in one call.
/// Used by the hidden `machine resize` backward-compat command.
pub fn resize_vm(
    name: &str,
    new_storage_gb: Option<u64>,
    new_overlay_gb: Option<u64>,
) -> smolvm::Result<()> {
    use smolvm::config::RecordState;
    use smolvm::db::SmolvmDb;

    let db = SmolvmDb::open()?;
    let record = db
        .get_vm(name)?
        .ok_or_else(|| smolvm::Error::vm_not_found(name))?
        .clone();

    let actual_state = record.actual_state();
    match actual_state {
        RecordState::Stopped | RecordState::Created => {}
        _ => {
            return Err(smolvm::Error::InvalidState {
                expected: "stopped".into(),
                actual: format!("{:?}", actual_state),
            });
        }
    }

    let changes = expand_disks(name, &record, new_storage_gb, new_overlay_gb)?;

    db.update_vm(name, |r| {
        if let Some(s) = new_storage_gb {
            r.storage_gb = Some(s);
        }
        if let Some(o) = new_overlay_gb {
            r.overlay_gb = Some(o);
        }
    })?;

    if changes.is_empty() {
        println!("No disk changes needed.");
    } else {
        println!("Resized machine '{}':", name);
        for c in &changes {
            println!("{}", c);
        }
        println!("Filesystem will expand on next boot.");
    }

    Ok(())
}

// ============================================================================
// Ephemeral VM Tracking
// ============================================================================

/// Clean up orphaned ephemeral VM records.
///
/// Called once at CLI startup. Scans for ephemeral records whose PID is no
/// longer alive and removes them. Fast path: if no ephemeral records exist,
/// this is a single DB read (~0.2ms).
pub fn cleanup_orphaned_ephemeral_vms() {
    let db = match SmolvmDb::open() {
        Ok(db) => db,
        Err(_) => return,
    };

    let vms = match db.list_vms() {
        Ok(vms) => vms,
        Err(_) => return,
    };

    for (name, record) in &vms {
        if !record.ephemeral {
            continue;
        }

        let is_orphan = match record.pid {
            Some(pid) => !smolvm::process::is_alive(pid),
            None => true, // No PID recorded — stale
        };

        if is_orphan {
            tracing::debug!(name = %name, pid = ?record.pid, "cleaning up orphaned ephemeral VM");
            let _ = db.remove_vm(name);
            let dir = smolvm::agent::vm_data_dir(name);
            if dir.exists() {
                let _ = std::fs::remove_dir_all(&dir);
            }
        }
    }
}
