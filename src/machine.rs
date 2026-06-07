//! Typed machine lifecycle service.
//!
//! The CLI, HTTP API, and Rust callers should adapt their inputs into these
//! request types and call a [`MachineService`] implementation. Keeping this as
//! the shared boundary prevents feature drift between adapters.

use std::{collections::BTreeMap, io::Write, path::PathBuf, time::Duration};

use smolvm_protocol::{ImageInfo, StorageStatus};

use crate::{
    agent::{
        machine_layers_cache_dir, resolve_disk_image, vm_data_dir, AgentClient, AgentManager,
        ExecEvent, LaunchFeatures, PullOptions, RunConfig,
    },
    config::{RecordState, RestartConfig, RestartPolicy, VmRecord},
    data::{
        network::PortMapping,
        resources::{
            validate_gpu_vram_mib, VmResources, DEFAULT_MICROVM_CPU_COUNT,
            DEFAULT_MICROVM_MEMORY_MIB,
        },
        storage::HostMount,
        validate_vm_name,
    },
    db::SmolvmDb,
    network::{validate_requested_network_backend, NetworkBackend},
    secrets::{self, ResolutionScope, SecretRef},
    storage::{expand_disk, DEFAULT_OVERLAY_SIZE_GIB, DEFAULT_STORAGE_SIZE_GIB},
    Error, Result,
};

macro_rules! machine_operation_catalog {
    ($macro:ident) => {
        $macro! {
            /// Create a machine record without starting it.
            Create => fn create(&self, request: CreateMachine) -> Result<MachineStatus>;
            /// Return one machine's status, or `None` when it does not exist.
            Status => fn status(&self, request: GetMachine) -> Result<Option<MachineStatus>>;
            /// List all known machines.
            List => fn list(&self, request: ListMachines) -> Result<Vec<MachineStatus>>;
            /// Start a machine.
            Start => fn start(&self, request: StartMachine) -> Result<MachineStatus>;
            /// Stop a machine.
            Stop => fn stop(&self, request: StopMachine) -> Result<MachineStatus>;
            /// Delete a machine.
            Delete => fn delete(&self, request: DeleteMachine) -> Result<()>;
            /// Fork a running forkable golden machine into a clone.
            Fork => fn fork(&self, request: ForkMachine) -> Result<MachineStatus>;
            /// Update a stopped machine's mutable configuration.
            Update => fn update(&self, request: UpdateMachine) -> Result<UpdateResult>;
            /// Execute a command and buffer stdout/stderr.
            Exec => fn exec(&self, request: ExecMachine) -> Result<ExecResult>;
            /// Execute a command and deliver stdout/stderr/exit events as they arrive.
            ExecStream => fn exec_stream(&self, request: ExecMachine, on_event: &mut dyn FnMut(ExecEvent)) -> Result<i32>;
            /// Execute a command using the process terminal for interactive I/O.
            ExecInteractive => fn exec_interactive(&self, request: ExecMachine) -> Result<i32>;
            /// Run an image-backed command inside an existing machine.
            Run => fn run(&self, request: RunMachine) -> Result<ExecResult>;
            /// Run a foreground or detached `machine run` style session.
            RunSession => fn run_session(&self, request: MachineRun, io: &mut dyn MachineRunIo) -> Result<MachineRunResult>;
            /// Monitor a machine's health and restart policy until stopped.
            Monitor => fn monitor(&self, request: MonitorMachine, on_event: &mut dyn FnMut(MonitorEvent), should_stop: &dyn Fn() -> bool) -> Result<MonitorResult>;
            /// Write bytes to a guest path.
            WriteFile => fn write_file(&self, request: WriteMachineFile) -> Result<FileTransfer>;
            /// Upload a host file to a guest path.
            UploadFile => fn upload_file(&self, request: UploadMachineFile) -> Result<FileTransfer>;
            /// Read a guest file into memory.
            ReadFile => fn read_file(&self, request: ReadMachineFile) -> Result<Vec<u8>>;
            /// Download a guest file to a host path.
            DownloadFile => fn download_file(&self, request: DownloadMachineFile) -> Result<FileTransfer>;
            /// Return OCI storage usage for a machine.
            StorageStatus => fn storage_status(&self, request: StorageStatusRequest) -> Result<StorageStatus>;
            /// List cached OCI images for a machine.
            ListImages => fn list_images(&self, request: ListMachineImages) -> Result<Vec<ImageInfo>>;
            /// Pull an OCI image into a machine's storage.
            PullImage => fn pull_image(&self, request: PullMachineImage) -> Result<ImageInfo>;
            /// Prune image/layer storage.
            PruneImages => fn prune_images(&self, request: PruneMachineImages) -> Result<PruneImagesResult>;
            /// Run a network connectivity diagnostic inside the machine.
            NetworkTest => fn network_test(&self, request: NetworkTestMachine) -> Result<serde_json::Value>;
            /// Return the host data directory for a machine.
            DataDir => fn data_dir(&self, request: DataDirMachine) -> Result<PathBuf>;
        }
    };
}

macro_rules! define_machine_operation_enum {
    ($( $(#[$meta:meta])* $variant:ident => fn $method:ident $args:tt -> $ret:ty; )*) => {
        /// Canonical machine service operation catalog.
        ///
        /// Adapter layers match exhaustively on this enum. Adding an operation
        /// to [`MachineService`] therefore forces CLI and HTTP bindings to make
        /// an explicit routing/support decision at compile time.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum MachineOperation {
            $(
                $(#[$meta])*
                $variant,
            )*
        }

        impl MachineOperation {
            /// All machine service operations.
            pub const ALL: &'static [MachineOperation] = &[
                $(MachineOperation::$variant,)*
            ];

            /// Rust method name for this operation.
            pub const fn method_name(self) -> &'static str {
                match self {
                    $(MachineOperation::$variant => stringify!($method),)*
                }
            }
        }
    };
}

macro_rules! define_machine_service_trait {
    ($( $(#[$meta:meta])* $variant:ident => fn $method:ident $args:tt -> $ret:ty; )*) => {
        /// Machine lifecycle operations shared by all adapters.
        pub trait MachineService {
            $(
                $(#[$meta])*
                fn $method $args -> $ret;
            )*
        }
    };
}

machine_operation_catalog!(define_machine_operation_enum);
machine_operation_catalog!(define_machine_service_trait);

/// Local DB/process-backed implementation of [`MachineService`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LocalMachineService {
    db: SmolvmDb,
}

impl LocalMachineService {
    /// Create a service backed by the default smolvm database.
    pub fn new() -> Result<Self> {
        Ok(Self {
            db: SmolvmDb::open()?,
        })
    }

    /// Create a service backed by an explicit database handle.
    pub fn with_db(db: SmolvmDb) -> Self {
        Self { db }
    }
}

impl MachineService for LocalMachineService {
    fn create(&self, request: CreateMachine) -> Result<MachineStatus> {
        let record = request.into_record()?;
        let reservation = CreateReservation::reserve(&self.db, &record.name)?;
        let create_result = (|| -> Result<()> {
            prepare_packed_layers_for_create(&record)?;
            reservation.commit(&record)
        })();
        if create_result.is_err() {
            rollback_created_data_dir(&record.name);
        }
        create_result?;
        Ok(MachineStatus {
            name: record.name.clone(),
            state: record.state.clone(),
            record,
        })
    }

    fn status(&self, request: GetMachine) -> Result<Option<MachineStatus>> {
        let Some(record) = self.db.get_vm(&request.name)? else {
            return Ok(None);
        };
        Ok(Some(status_from_record(&request.name, record)))
    }

    fn list(&self, _request: ListMachines) -> Result<Vec<MachineStatus>> {
        self.db
            .list_vms()?
            .into_iter()
            .map(|(name, record)| Ok(status_from_record(&name, record)))
            .collect()
    }

    fn start(&self, request: StartMachine) -> Result<MachineStatus> {
        let mut record = get_record(&self.db, &request.name)?;
        match crate::agent::state_probe::resolve_state(&request.name, &record) {
            RecordState::Running => return Ok(status_from_record(&request.name, record)),
            RecordState::Unreachable => {
                crate::agent::state_probe::recover_unreachable_machine(&record);
                record = get_record(&self.db, &request.name)?;
            }
            RecordState::Stopped | RecordState::Created | RecordState::Failed => {}
        }

        let manager =
            AgentManager::for_vm_with_sizes(&request.name, record.storage_gb, record.overlay_gb)
                .map_err(|error| Error::agent("create agent manager", error.to_string()))?;

        let features = launch_features(&request, &record)?;
        manager
            .ensure_running_with_full_config(
                record.host_mounts(),
                record.port_mappings(),
                record.vm_resources(),
                features,
            )
            .map_err(|error| Error::agent("start machine", error.to_string()))?;

        let pid = manager.child_pid();
        let mut client = AgentClient::connect_with_retry(manager.vsock_socket())?;
        let env = record_env_with_secrets(&record)?;

        if !record.init_completed {
            if request.snapshot_dir.is_none() {
                run_first_start(
                    &self.db,
                    &request.name,
                    &mut record,
                    &mut client,
                    &env,
                    &request,
                )?;
            }
            if !record.init.is_empty() || record.image.is_some() {
                let _ = self.db.update_vm(&request.name, |r| {
                    r.init_completed = true;
                });
            }
        }

        if request.snapshot_dir.is_none() {
            launch_workload(&request.name, &record, &mut client, env)?;
        }

        mark_running(&self.db, &request.name, pid)?;
        manager.detach();
        Ok(self
            .status(GetMachine::new(request.name))?
            .expect("machine exists after start"))
    }

    fn stop(&self, request: StopMachine) -> Result<MachineStatus> {
        let record = get_record(&self.db, &request.name)?;
        match crate::agent::state_probe::resolve_state(&request.name, &record) {
            RecordState::Running => {
                let manager = AgentManager::for_vm(&request.name)
                    .map_err(|error| Error::agent("create agent manager", error.to_string()))?;
                manager.stop()?;
            }
            RecordState::Unreachable => {
                crate::agent::state_probe::recover_unreachable_machine(&record);
            }
            RecordState::Stopped | RecordState::Created | RecordState::Failed => {}
        }
        if record.source_smolmachine.is_some() {
            smolvm_pack::extract::force_detach_layers_volume(&machine_layers_cache_dir(
                &request.name,
            ));
        }
        mark_stopped(&self.db, &request.name)?;
        Ok(self
            .status(GetMachine::new(request.name))?
            .expect("machine exists after stop"))
    }

    fn delete(&self, request: DeleteMachine) -> Result<()> {
        let record = get_record(&self.db, &request.name)?;
        let dependent_clones = self.db.dependent_clones(&request.name)?;
        if !dependent_clones.is_empty() && !request.break_dependent_clones {
            return Err(Error::agent(
                "delete",
                format!(
                    "machine '{}' is the fork base for {} clone(s) ({})",
                    request.name,
                    dependent_clones.len(),
                    dependent_clones.join(", ")
                ),
            ));
        }

        if crate::agent::state_probe::resolve_state(&request.name, &record) == RecordState::Running
            || crate::agent::state_probe::resolve_state(&request.name, &record)
                == RecordState::Unreachable
        {
            self.stop(StopMachine::new(request.name.clone()))?;
        }

        self.db.remove_vm(&request.name)?;
        smolvm_pack::extract::force_detach_layers_volume(&machine_layers_cache_dir(&request.name));
        let data_dir = vm_data_dir(&request.name);
        match std::fs::remove_dir_all(&data_dir) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(Error::storage(
                "delete machine data",
                format!("{}: {error}", data_dir.display()),
            )),
        }
    }

    fn fork(&self, request: ForkMachine) -> Result<MachineStatus> {
        validate_vm_name(&request.clone, "clone name")
            .map_err(|reason| Error::config("fork machine", reason))?;
        if self.db.get_vm(&request.clone)?.is_some() {
            return Err(Error::agent(
                "fork",
                format!("machine '{}' already exists", request.clone),
            ));
        }

        let golden = get_record(&self.db, &request.golden)?;
        let control_socket = control_socket_path(&request.golden);
        let status = control_socket_cmd(&control_socket, "STATUS").map_err(|error| {
            Error::agent(
                "fork",
                format!(
                    "golden '{}' is not running forkable ({error}); start it with forkable=true",
                    request.golden
                ),
            )
        })?;
        if !status.starts_with("OK") {
            return Err(Error::agent(
                "fork",
                format!("golden '{}' is not ready to fork: {status}", request.golden),
            ));
        }

        let clone_dir = vm_data_dir(&request.clone);
        let snapshot_dir = clone_dir.join("snapshot");
        std::fs::create_dir_all(&snapshot_dir)
            .map_err(|error| Error::agent("create clone dir", error.to_string()))?;

        let mut clone = golden.clone();
        clone.name = request.clone.clone();
        clone.pid = None;
        clone.pid_start_time = None;
        clone.golden = Some(request.golden.clone());
        clone.ports = clone_ports(&golden, &request.ports);
        self.db.insert_vm(&request.clone, &clone)?;

        let result = (|| -> Result<()> {
            let reply =
                control_socket_cmd(&control_socket, &format!("FORK {}", snapshot_dir.display()))?;
            if !reply.starts_with("OK") {
                return Err(Error::agent("fork", format!("golden FORK failed: {reply}")));
            }
            clone_disks(&request.golden, &request.clone)?;
            self.start(StartMachine {
                name: request.clone.clone(),
                forkable: false,
                proxy: None,
                no_proxy: None,
                snapshot_dir: Some(snapshot_dir),
            })?;
            rejuvenate_clone(&request.clone);
            Ok(())
        })();

        if let Err(error) = result {
            let _ = self.db.remove_vm(&request.clone);
            let _ = std::fs::remove_dir_all(&clone_dir);
            return Err(error);
        }

        Ok(self
            .status(GetMachine::new(request.clone))?
            .expect("clone exists after fork"))
    }

    fn update(&self, request: UpdateMachine) -> Result<UpdateResult> {
        let record = get_record(&self.db, &request.name)?;
        let actual_state = crate::agent::state_probe::resolve_state(&request.name, &record);
        match actual_state {
            RecordState::Stopped | RecordState::Created => {}
            _ => {
                return Err(Error::InvalidState {
                    expected: "stopped".into(),
                    actual: actual_state.to_string(),
                });
            }
        }

        for (key, _) in &request.set_env {
            if key.is_empty() {
                return Err(Error::config("update", "env key must not be empty"));
            }
        }

        let proposed_resources = crate::agent::VmResources {
            cpus: request.cpus.unwrap_or(record.cpus),
            memory_mib: request.memory_mib.unwrap_or(record.mem),
            network: if request.disable_network {
                false
            } else {
                request.enable_network || record.network
            },
            network_backend: request.network_backend.or(record.network_backend),
            gpu: if request.disable_gpu {
                false
            } else {
                request.enable_gpu || record.gpu.unwrap_or(false)
            },
            gpu_vram_mib: request.gpu_vram_mib.or(record.gpu_vram_mib),
            storage_gib: request.storage_gb.or(record.storage_gb),
            overlay_gib: request.overlay_gb.or(record.overlay_gb),
            allowed_cidrs: if request.disable_network || request.clear_allowed_cidrs {
                None
            } else {
                request
                    .allowed_cidrs
                    .clone()
                    .or(record.allowed_cidrs.clone())
            },
        };
        proposed_resources.validate()?;
        validate_requested_network_backend(
            &proposed_resources,
            request
                .dns_filter_hosts
                .as_deref()
                .or(record.dns_filter_hosts.as_deref()),
            proposed_ports(&record, &request)?.len(),
        )?;

        let gpu_vram_mib = validate_gpu_vram_mib(request.gpu_vram_mib)
            .map_err(|error| Error::config("update", format!("gpu_vram: {error}")))?;

        let disk_changes = expand_machine_disks(
            &request.name,
            &record,
            request.storage_gb,
            request.overlay_gb,
        )?;

        let mut changes = disk_changes;
        let updated = self
            .db
            .update_vm(&request.name, |r| {
                apply_update(r, &request, gpu_vram_mib, &mut changes);
            })?
            .ok_or_else(|| Error::vm_not_found(&request.name))?;

        Ok(UpdateResult {
            status: status_from_record(&request.name, updated),
            changes,
        })
    }

    fn exec(&self, request: ExecMachine) -> Result<ExecResult> {
        let mut prepared = self.prepare_exec(&request)?;
        if request.background {
            let pid = if let Some(image) = prepared.record.image.clone() {
                let config = prepared.image_run_config(image, request.command)?;
                prepared.client.run_background(config)?
            } else {
                prepared.client.vm_exec_background(
                    request.command,
                    prepared.env,
                    prepared.workdir,
                )?
            };
            return Ok(ExecResult {
                exit_code: 0,
                stdout: format!("pid={pid}\n").into_bytes(),
                stderr: Vec::new(),
            });
        }

        if let Some(image) = prepared.record.image.clone() {
            let config = prepared.image_run_config(image, request.command)?;
            let (exit_code, stdout, stderr) = prepared.client.run_non_interactive(config)?;
            Ok(ExecResult {
                exit_code,
                stdout,
                stderr,
            })
        } else {
            let (exit_code, stdout, stderr) = prepared.client.vm_exec(
                request.command,
                prepared.env,
                prepared.workdir,
                request.timeout,
                request.stdin,
            )?;
            Ok(ExecResult {
                exit_code,
                stdout,
                stderr,
            })
        }
    }

    fn exec_stream(
        &self,
        request: ExecMachine,
        on_event: &mut dyn FnMut(ExecEvent),
    ) -> Result<i32> {
        let mut prepared = self.prepare_exec(&request)?;
        let mut exit_code = 0;
        let mut forward = |event: ExecEvent| {
            if let ExecEvent::Exit(code) = event {
                exit_code = code;
            }
            on_event(event);
        };
        if let Some(image) = prepared.record.image.clone() {
            let config = prepared.image_run_config(image, request.command)?;
            prepared.client.run_streaming_with(config, &mut forward)?;
        } else {
            prepared.client.vm_exec_streaming_with(
                request.command,
                prepared.env,
                prepared.workdir,
                request.timeout,
                &mut forward,
            )?;
        }
        Ok(exit_code)
    }

    fn exec_interactive(&self, request: ExecMachine) -> Result<i32> {
        let mut prepared = self.prepare_exec(&request)?;
        if let Some(image) = prepared.record.image.clone() {
            let config = prepared
                .image_run_config(image, request.command)?
                .with_tty(request.tty);
            prepared.client.run_interactive(config)
        } else {
            prepared.client.vm_exec_interactive(
                request.command,
                prepared.env,
                prepared.workdir,
                request.timeout,
                request.tty,
            )
        }
    }

    fn run(&self, request: RunMachine) -> Result<ExecResult> {
        let (record, mut client) = self.client_for_operation(
            &request.name,
            request.start_if_needed,
            request.trace_id.clone(),
        )?;
        let env = request_env(&request.env, &request.secret_refs, request.secret_scope)?;
        let mounts = record_mounts_to_runconfig_bindings(&record.mounts);
        let config = RunConfig::new(request.image, request.command)
            .with_env(env)
            .with_workdir(request.workdir)
            .with_mounts(mounts)
            .with_timeout(request.timeout)
            .with_persistent_overlay(request.persistent_overlay_id);
        let (exit_code, stdout, stderr) = client.run_non_interactive(config)?;
        Ok(ExecResult {
            exit_code,
            stdout,
            stderr,
        })
    }

    fn run_session(
        &self,
        request: MachineRun,
        io: &mut dyn MachineRunIo,
    ) -> Result<MachineRunResult> {
        run_machine_session(self, request, io)
    }

    fn monitor(
        &self,
        request: MonitorMachine,
        on_event: &mut dyn FnMut(MonitorEvent),
        should_stop: &dyn Fn() -> bool,
    ) -> Result<MonitorResult> {
        monitor_machine(self, request, on_event, should_stop)
    }

    fn write_file(&self, request: WriteMachineFile) -> Result<FileTransfer> {
        let (record, mut client) = self.client_for_operation(
            &request.name,
            request.start_if_needed,
            request.trace_id.clone(),
        )?;
        prepare_image_overlay_for_file_ops(&record, &mut client, &request.name)?;
        let size = request.data.len() as u64;
        client.write_file(&request.guest_path, &request.data, request.mode)?;
        Ok(FileTransfer { bytes: size })
    }

    fn upload_file(&self, request: UploadMachineFile) -> Result<FileTransfer> {
        let (record, mut client) = self.client_for_operation(
            &request.name,
            request.start_if_needed,
            request.trace_id.clone(),
        )?;
        prepare_image_overlay_for_file_ops(&record, &mut client, &request.name)?;
        let file = std::fs::File::open(&request.local_path).map_err(|error| {
            Error::agent(
                "read local file",
                format!("{}: {error}", request.local_path.display()),
            )
        })?;
        let size = file
            .metadata()
            .map_err(|error| {
                Error::agent(
                    "stat local file",
                    format!("{}: {error}", request.local_path.display()),
                )
            })?
            .len();
        client.write_file_from_reader(&request.guest_path, file, size, request.mode)?;
        Ok(FileTransfer { bytes: size })
    }

    fn read_file(&self, request: ReadMachineFile) -> Result<Vec<u8>> {
        let (record, mut client) = self.client_for_operation(
            &request.name,
            request.start_if_needed,
            request.trace_id.clone(),
        )?;
        prepare_image_overlay_for_file_ops(&record, &mut client, &request.name)?;
        client.read_file(&request.guest_path)
    }

    fn download_file(&self, request: DownloadMachineFile) -> Result<FileTransfer> {
        let (record, mut client) = self.client_for_operation(
            &request.name,
            request.start_if_needed,
            request.trace_id.clone(),
        )?;
        prepare_image_overlay_for_file_ops(&record, &mut client, &request.name)?;
        let bytes = client.read_file_to_path(&request.guest_path, &request.local_path, |_| {})?;
        Ok(FileTransfer { bytes })
    }

    fn storage_status(&self, request: StorageStatusRequest) -> Result<StorageStatus> {
        let (_, mut client) = self.client_for_operation(
            &request.name,
            request.start_if_needed,
            request.trace_id.clone(),
        )?;
        client.storage_status()
    }

    fn list_images(&self, request: ListMachineImages) -> Result<Vec<ImageInfo>> {
        let state = self
            .status(GetMachine::new(request.name.clone()))?
            .ok_or_else(|| Error::vm_not_found(&request.name))?
            .state;
        if state != RecordState::Running && !request.start_if_needed {
            if request.empty_when_stopped {
                return Ok(Vec::new());
            }
            return Err(Error::InvalidState {
                expected: "running".into(),
                actual: state.to_string(),
            });
        }
        let started_by_service = state != RecordState::Running && request.start_if_needed;
        let (_, mut client) = self.client_for_operation(
            &request.name,
            request.start_if_needed,
            request.trace_id.clone(),
        )?;
        let result = client.list_images();
        if started_by_service && request.stop_after_start {
            let _ = self.stop(StopMachine::new(request.name));
        }
        result
    }

    fn pull_image(&self, request: PullMachineImage) -> Result<ImageInfo> {
        if request.image.is_empty() {
            return Err(Error::config(
                "pull image",
                "image reference cannot be empty",
            ));
        }
        let (_, mut client) = self.client_for_operation(
            &request.name,
            request.start_if_needed,
            request.trace_id.clone(),
        )?;
        let mut options = PullOptions::new().use_registry_config(true);
        if let Some(platform) = request.oci_platform {
            options = options.oci_platform(platform);
        }
        if let Some(proxy) = request.proxy {
            options = options.proxy(proxy);
        }
        if let Some(no_proxy) = request.no_proxy {
            options = options.no_proxy(no_proxy);
        }
        client.pull(&request.image, options)
    }

    fn prune_images(&self, request: PruneMachineImages) -> Result<PruneImagesResult> {
        let state = self
            .status(GetMachine::new(request.name.clone()))?
            .ok_or_else(|| Error::vm_not_found(&request.name))?
            .state;
        if request.all && state == RecordState::Running {
            return Err(Error::agent(
                "prune",
                format!(
                    "cannot prune all images while machine '{}' is running",
                    request.name
                ),
            ));
        }
        let started_by_service = state != RecordState::Running;
        let (_, mut client) = self.client_for_operation(&request.name, true, request.trace_id)?;
        let images = if request.all {
            client.list_images()?
        } else {
            Vec::new()
        };
        let freed_bytes = if request.all && request.dry_run {
            images.iter().map(|i| i.size).sum()
        } else {
            client.garbage_collect(request.dry_run, request.all)?
        };
        if started_by_service && request.stop_after_start {
            let _ = self.stop(StopMachine::new(request.name.clone()));
        }
        Ok(PruneImagesResult {
            freed_bytes,
            removed_images: if request.all { images.len() } else { 0 },
            dry_run: request.dry_run,
        })
    }

    fn network_test(&self, request: NetworkTestMachine) -> Result<serde_json::Value> {
        let (_, mut client) = self.client_for_operation(
            &request.name,
            request.start_if_needed,
            request.trace_id.clone(),
        )?;
        client.network_test(&request.url)
    }

    fn data_dir(&self, request: DataDirMachine) -> Result<PathBuf> {
        if request.require_existing {
            let _ = get_record(&self.db, &request.name)?;
        }
        Ok(vm_data_dir(&request.name))
    }
}

impl LocalMachineService {
    fn client_for_operation(
        &self,
        name: &str,
        start_if_needed: bool,
        trace_id: Option<String>,
    ) -> Result<(VmRecord, AgentClient)> {
        let mut record = get_record(&self.db, name)?;
        let state = crate::agent::state_probe::resolve_state(name, &record);
        if state != RecordState::Running && start_if_needed {
            self.start(StartMachine::new(name.to_string()))?;
            record = get_record(&self.db, name)?;
        }

        let manager = AgentManager::for_vm_with_sizes(name, record.storage_gb, record.overlay_gb)
            .map_err(|error| Error::agent("create agent manager", error.to_string()))?;
        if manager.try_connect_existing().is_none() {
            return Err(Error::InvalidState {
                expected: "running".into(),
                actual: state.to_string(),
            });
        }
        let mut client = AgentClient::connect_with_retry(manager.vsock_socket())?;
        if let Some(trace_id) = trace_id {
            client.set_trace_id(trace_id);
        }
        manager.detach();
        Ok((record, client))
    }

    fn prepare_exec(&self, request: &ExecMachine) -> Result<PreparedExec> {
        let (record, client) = self.client_for_operation(
            &request.name,
            request.start_if_needed,
            request.trace_id.clone(),
        )?;
        let mut env = if request.include_record_env {
            record_env_with_secrets(&record)?
        } else {
            Vec::new()
        };
        env.extend(request_env(
            &[],
            &request.secret_refs,
            request.secret_scope,
        )?);
        env = merge_env_overrides(&env, &request.env);
        let workdir = request.workdir.clone().or_else(|| record.workdir.clone());
        Ok(PreparedExec {
            record,
            client,
            env,
            workdir,
            timeout: request.timeout,
            tty: request.tty,
        })
    }
}

fn run_machine_session(
    service: &LocalMachineService,
    request: MachineRun,
    io: &mut dyn MachineRunIo,
) -> Result<MachineRunResult> {
    request.resources.validate()?;
    validate_requested_network_backend(
        &request.resources,
        request.dns_filter_hosts.as_deref(),
        request.ports.len(),
    )?;
    validate_ports(&request.ports)?;
    if request.detached
        && !request.allow_existing_record
        && service.db.get_vm(&request.name)?.is_some()
    {
        return Err(Error::config(
            "machine run",
            format!("machine '{}' already exists", request.name),
        ));
    }

    io.starting(&MachineRunStarting {
        name: request.name.clone(),
        detached: request.detached,
    });

    let manager = AgentManager::for_vm_with_sizes(
        &request.name,
        request.resources.storage_gib,
        request.resources.overlay_gib,
    )
    .map_err(|error| Error::agent("create agent manager", error.to_string()))?;

    let ssh_agent_socket = if request.ssh_agent {
        Some(
            std::env::var_os("SSH_AUTH_SOCK")
                .ok_or_else(|| Error::config("ssh-agent", "SSH_AUTH_SOCK is not set"))?
                .into(),
        )
    } else {
        None
    };
    let features = LaunchFeatures {
        ssh_agent_socket,
        dns_filter_hosts: request.dns_filter_hosts.clone(),
        ..Default::default()
    };

    let freshly_started = manager
        .ensure_running_with_full_config(
            request.mounts.clone(),
            request.ports.clone(),
            request.resources.clone(),
            features,
        )
        .map_err(|error| Error::agent("start machine", error.to_string()))?;

    let pid = manager.child_pid();
    if !request.detached {
        register_ephemeral_run(&service.db, &request, pid);
    }

    let mut client = AgentClient::connect_with_retry(manager.vsock_socket())?;
    let sigint_guard = if request.kill_on_sigint {
        pid.map(crate::process::SigintGuard::new)
    } else {
        None
    };

    let result = run_machine_session_started(
        service,
        &request,
        io,
        &manager,
        &mut client,
        freshly_started,
        sigint_guard,
    );

    if !request.detached {
        deregister_ephemeral_run(&service.db, &request.name);
        manager.kill();
        manager.cleanup_data_dir();
    } else if result.is_err() && freshly_started {
        let _ = manager.stop();
    }

    result
}

fn run_machine_session_started(
    service: &LocalMachineService,
    request: &MachineRun,
    io: &mut dyn MachineRunIo,
    manager: &AgentManager,
    client: &mut AgentClient,
    freshly_started: bool,
    sigint_guard: Option<crate::process::SigintGuard>,
) -> Result<MachineRunResult> {
    let image_info = if let Some(image) = request.image.as_ref() {
        let image_for_progress = image.clone();
        let pull = client.pull_with_registry_config_and_progress(
            image,
            request.oci_platform.as_deref(),
            request.proxy.as_deref(),
            request.no_proxy.as_deref(),
            |current, total, layer| {
                io.pull_progress(&MachineRunPullProgress {
                    image: image_for_progress.clone(),
                    current,
                    total,
                    layer: layer.to_string(),
                });
            },
        );
        match pull {
            Ok(info) => Some(info),
            Err(error) if !request.resources.network => {
                return Err(Error::agent(
                    "pull image",
                    format!(
                        "{}\n\nHint: networking is disabled. Enable networking before pulling image '{}'.",
                        error, image
                    ),
                ));
            }
            Err(error) => return Err(error),
        }
    } else {
        None
    };

    let env = request_env(&request.env, &request.secret_refs, request.secret_scope)?;
    if freshly_started && !request.init.is_empty() {
        run_session_init(
            client,
            &request.init,
            SessionInitContext {
                image: request.image.as_deref(),
                image_info: image_info.as_ref(),
                env: &env,
                workdir: request.workdir.as_deref(),
                user: request.user.as_deref(),
                mounts: &request.mounts,
                overlay_id: &request.name,
            },
        )?;
    }

    let command = resolve_session_command(request, image_info.as_ref());
    let mount_bindings = host_mounts_to_runconfig_bindings(&request.mounts);

    if request.detached {
        if let Some(image) = request.image.as_ref() {
            let defaults = resolve_image_runtime_defaults(
                image_info.as_ref(),
                &env,
                request.workdir.as_deref(),
                request.user.as_deref(),
            );
            let config = RunConfig::new(image, command.clone())
                .with_env(defaults.env.clone())
                .with_workdir(defaults.workdir.clone())
                .with_user(defaults.user.clone())
                .with_mounts(mount_bindings)
                .with_persistent_overlay(Some(request.name.clone()));
            client.run_container_detached(config)?;
            persist_detached_run(
                &service.db,
                request,
                DetachedRunRecord {
                    pid: manager.child_pid(),
                    image: Some(image.clone()),
                    entrypoint: Vec::new(),
                    cmd: command,
                    env: defaults.env,
                    workdir: defaults.workdir,
                    user: defaults.user,
                },
            )?;
        } else {
            let is_idle = command.is_empty()
                || command
                    == crate::DEFAULT_IDLE_CMD
                        .iter()
                        .map(|value| value.to_string())
                        .collect::<Vec<_>>();
            if !is_idle {
                let _pid =
                    client.vm_exec_background(command, env.clone(), request.workdir.clone())?;
            }
            persist_detached_run(
                &service.db,
                request,
                DetachedRunRecord {
                    pid: manager.child_pid(),
                    image: None,
                    entrypoint: request.entrypoint.clone(),
                    cmd: request.cmd.clone(),
                    env: request.env.clone(),
                    workdir: request.workdir.clone(),
                    user: request.user.clone(),
                },
            )?;
        }
        if let Some(guard) = sigint_guard {
            guard.disarm();
        }
        manager.detach();
        return Ok(MachineRunResult::Detached {
            name: request.name.clone(),
            pid: manager.child_pid(),
        });
    }

    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let exit_code = if let Some(image) = request.image.as_ref() {
        let defaults = resolve_image_runtime_defaults(
            image_info.as_ref(),
            &env,
            request.workdir.as_deref(),
            request.user.as_deref(),
        );
        if request.interactive || request.tty {
            if let Some(guard) = sigint_guard {
                guard.disarm();
            }
            let config = RunConfig::new(image, command)
                .with_env(defaults.env)
                .with_workdir(defaults.workdir)
                .with_user(defaults.user)
                .with_mounts(mount_bindings)
                .with_timeout(request.timeout)
                .with_tty(request.tty);
            client.run_interactive(config)?
        } else {
            let config = RunConfig::new(image, command)
                .with_env(defaults.env)
                .with_workdir(defaults.workdir)
                .with_user(defaults.user)
                .with_mounts(mount_bindings)
                .with_timeout(request.timeout);
            let (code, out, err) = client.run_non_interactive(config)?;
            io.stdout(&out);
            io.stderr(&err);
            stdout = out;
            stderr = err;
            code
        }
    } else if request.interactive || request.tty {
        if let Some(guard) = sigint_guard {
            guard.disarm();
        }
        client.vm_exec_interactive(
            command,
            env,
            request.workdir.clone(),
            request.timeout,
            request.tty,
        )?
    } else {
        let cmd0 = command.first().cloned().unwrap_or_default();
        let (code, out, err) = client
            .vm_exec(command, env, request.workdir.clone(), request.timeout, None)
            .map_err(|error| enrich_bare_run_error(error, &cmd0))?;
        io.stdout(&out);
        io.stderr(&err);
        stdout = out;
        stderr = err;
        code
    };

    Ok(MachineRunResult::Foreground {
        exit_code,
        stdout,
        stderr,
    })
}

fn resolve_session_command(request: &MachineRun, image_info: Option<&ImageInfo>) -> Vec<String> {
    if !request.command.is_empty() {
        request.command.clone()
    } else if !request.entrypoint.is_empty() || !request.cmd.is_empty() {
        let mut command = request.entrypoint.clone();
        command.extend(request.cmd.clone());
        command
    } else if let Some(info) = image_info {
        let mut command = info.entrypoint.clone();
        command.extend(info.cmd.clone());
        if !command.is_empty() {
            command
        } else if request.detached {
            crate::DEFAULT_IDLE_CMD
                .iter()
                .map(|value| value.to_string())
                .collect()
        } else {
            vec![crate::DEFAULT_SHELL_CMD.to_string()]
        }
    } else if request.detached {
        crate::DEFAULT_IDLE_CMD
            .iter()
            .map(|value| value.to_string())
            .collect()
    } else {
        vec![crate::DEFAULT_SHELL_CMD.to_string()]
    }
}

struct SessionInitContext<'a> {
    image: Option<&'a str>,
    image_info: Option<&'a ImageInfo>,
    env: &'a [(String, String)],
    workdir: Option<&'a str>,
    user: Option<&'a str>,
    mounts: &'a [HostMount],
    overlay_id: &'a str,
}

fn run_session_init(
    client: &mut AgentClient,
    init: &[String],
    context: SessionInitContext<'_>,
) -> Result<()> {
    for (i, cmd) in init.iter().enumerate() {
        let argv = vec!["sh".to_string(), "-c".to_string(), cmd.clone()];
        let (code, stdout, stderr) = if let Some(image) = context.image {
            let defaults = resolve_image_runtime_defaults(
                context.image_info,
                context.env,
                context.workdir,
                context.user,
            );
            let config = RunConfig::new(image, argv)
                .with_env(defaults.env)
                .with_workdir(defaults.workdir)
                .with_user(defaults.user)
                .with_mounts(host_mounts_to_runconfig_bindings(context.mounts))
                .with_persistent_overlay(Some(context.overlay_id.to_string()));
            client.run_non_interactive(config)?
        } else {
            client.vm_exec(
                argv,
                context.env.to_vec(),
                context.workdir.map(str::to_string),
                None,
                None,
            )?
        };
        if code != 0 {
            return Err(Error::agent(
                "run init",
                format!(
                    "init[{i}] failed (exit {code}): {}; stdout: {}",
                    String::from_utf8_lossy(&stderr).trim(),
                    String::from_utf8_lossy(&stdout).trim()
                ),
            ));
        }
    }
    Ok(())
}

struct DetachedRunRecord {
    pid: Option<i32>,
    image: Option<String>,
    entrypoint: Vec<String>,
    cmd: Vec<String>,
    env: Vec<(String, String)>,
    workdir: Option<String>,
    user: Option<String>,
}

fn persist_detached_run(
    db: &SmolvmDb,
    request: &MachineRun,
    detached: DetachedRunRecord,
) -> Result<()> {
    let mount_tuples: Vec<(String, String, bool)> = request
        .mounts
        .iter()
        .map(HostMount::to_storage_tuple)
        .collect();
    let mut record = VmRecord::new(
        request.name.clone(),
        request.resources.cpus,
        request.resources.memory_mib,
        mount_tuples,
        PortMapping::to_tuples(&request.ports),
        request.resources.network,
    );
    record.state = RecordState::Running;
    record.pid = detached.pid;
    record.pid_start_time = detached.pid.and_then(crate::process::process_start_time);
    record.network_backend = request.resources.network_backend;
    record.storage_gb = request.resources.storage_gib;
    record.overlay_gb = request.resources.overlay_gib;
    record.allowed_cidrs = request.resources.allowed_cidrs.clone();
    record.init = request.init.clone();
    record.init_completed = false;
    record.env = detached.env;
    record.secret_refs = request.secret_refs.clone();
    record.workdir = detached.workdir;
    record.user = detached.user;
    record.image = detached.image;
    record.entrypoint = detached.entrypoint;
    record.cmd = detached.cmd;
    record.ssh_agent = request.ssh_agent;
    record.dns_filter_hosts = request.dns_filter_hosts.clone();
    record.gpu = request.resources.gpu.then_some(true);
    record.gpu_vram_mib = request.resources.gpu_vram_mib;

    if db.get_vm(&request.name)?.is_some() {
        db.update_vm(&request.name, |slot| {
            *slot = record.clone();
        })?
        .ok_or_else(|| Error::vm_not_found(&request.name))?;
    } else {
        db.insert_vm(&request.name, &record)?;
    }
    Ok(())
}

fn register_ephemeral_run(db: &SmolvmDb, request: &MachineRun, pid: Option<i32>) {
    let mut record = VmRecord::new(
        request.name.clone(),
        request.resources.cpus,
        request.resources.memory_mib,
        Vec::new(),
        Vec::new(),
        request.resources.network,
    );
    record.ephemeral = true;
    record.state = RecordState::Running;
    record.pid = pid;
    record.image = request.image.clone();
    if let Err(error) = db.insert_vm(&request.name, &record) {
        tracing::debug!(error = %error, name = %request.name, "failed to register ephemeral VM");
    }
}

fn deregister_ephemeral_run(db: &SmolvmDb, name: &str) {
    if let Err(error) = db.remove_vm(name) {
        tracing::debug!(error = %error, name, "failed to deregister ephemeral VM");
    }
}

fn host_mounts_to_runconfig_bindings(mounts: &[HostMount]) -> Vec<(String, String, bool)> {
    mounts
        .iter()
        .enumerate()
        .map(|(index, mount)| {
            (
                HostMount::mount_tag(index),
                mount.target.to_string_lossy().into_owned(),
                mount.read_only,
            )
        })
        .collect()
}

fn enrich_bare_run_error(error: Error, cmd0: &str) -> Error {
    let msg = error.to_string();
    if !cmd0.is_empty()
        && (msg.contains("No such file or directory") || msg.contains("os error 2"))
        && !cmd0.starts_with('/')
        && !cmd0.starts_with('.')
    {
        Error::agent(
            "vm exec",
            format!(
                "{msg}\n\nNote: '{cmd0}' was not found in the VM. If you meant to run a container image, set an image for the run request."
            ),
        )
    } else {
        error
    }
}

fn monitor_machine(
    service: &LocalMachineService,
    request: MonitorMachine,
    on_event: &mut dyn FnMut(MonitorEvent),
    should_stop: &dyn Fn() -> bool,
) -> Result<MonitorResult> {
    let mut record = get_record(&service.db, &request.name)?;
    let mut restart = record.restart.clone();
    if let Some(policy) = request.restart_policy.clone() {
        restart.policy = policy;
    }
    let health_cmd = request
        .health_cmd
        .clone()
        .or_else(|| record.health_cmd.clone());
    let health_timeout = Duration::from_secs(
        record
            .health_timeout_secs
            .unwrap_or(request.health_timeout.as_secs()),
    );
    let health_retries = record.health_retries.unwrap_or(request.health_retries);
    let interval = Duration::from_secs(
        record
            .health_interval_secs
            .unwrap_or(request.interval.as_secs()),
    );
    let startup_grace = record
        .health_startup_grace_secs
        .map(Duration::from_secs)
        .unwrap_or(Duration::ZERO);

    let manager = AgentManager::for_vm(&request.name)
        .map_err(|error| Error::agent("create agent manager", error.to_string()))?;
    if !manager.is_process_alive() {
        on_event(MonitorEvent::Starting {
            name: request.name.clone(),
        });
        service.start(StartMachine::new(request.name.clone()))?;
        record = get_record(&service.db, &request.name)?;
        restart = record.restart.clone();
        if let Some(policy) = request.restart_policy.clone() {
            restart.policy = policy;
        }
    }

    on_event(MonitorEvent::Monitoring {
        name: request.name.clone(),
        policy: restart.policy.clone(),
        interval_secs: interval.as_secs(),
        health: health_cmd.as_ref().map(|command| MonitorHealthConfig {
            command: command.clone(),
            timeout_secs: health_timeout.as_secs(),
            retries: health_retries,
        }),
    });

    let mut consecutive_health_failures = 0;
    let mut last_check = std::time::Instant::now();
    let mut last_start = std::time::Instant::now();

    loop {
        std::thread::sleep(interval);
        if should_stop() {
            break;
        }

        let elapsed = last_check.elapsed();
        last_check = std::time::Instant::now();
        if elapsed > interval * 3 {
            on_event(MonitorEvent::SuspendDetected {
                sleep_secs: elapsed.as_secs().saturating_sub(interval.as_secs()),
            });
            consecutive_health_failures = 0;
            continue;
        }

        let manager = match AgentManager::for_vm(&request.name) {
            Ok(manager) => manager,
            Err(_) => continue,
        };

        if manager.is_process_alive() {
            if !startup_grace.is_zero() && last_start.elapsed() < startup_grace {
                continue;
            }
            if let Some(command) = health_cmd.as_ref() {
                match AgentClient::connect_with_short_timeout(manager.vsock_socket()) {
                    Ok(mut client) => match client.vm_exec(
                        command.clone(),
                        Vec::new(),
                        None,
                        Some(health_timeout),
                        None,
                    ) {
                        Ok((0, _, _)) => {
                            if consecutive_health_failures > 0 {
                                on_event(MonitorEvent::HealthRecovered);
                            }
                            consecutive_health_failures = 0;
                        }
                        Ok((exit_code, _, stderr)) => {
                            consecutive_health_failures += 1;
                            on_event(MonitorEvent::HealthFailed {
                                exit_code,
                                consecutive: consecutive_health_failures,
                                retries: health_retries,
                                stderr: String::from_utf8_lossy(&stderr).trim().to_string(),
                            });
                        }
                        Err(error) => {
                            consecutive_health_failures += 1;
                            on_event(MonitorEvent::HealthError {
                                consecutive: consecutive_health_failures,
                                retries: health_retries,
                                error: error.to_string(),
                            });
                        }
                    },
                    Err(_) => {
                        consecutive_health_failures += 1;
                        on_event(MonitorEvent::AgentUnreachable {
                            consecutive: consecutive_health_failures,
                            retries: health_retries,
                        });
                    }
                }
                if consecutive_health_failures >= health_retries {
                    on_event(MonitorEvent::UnhealthyStopping);
                    let _ = service.stop(StopMachine::new(request.name.clone()));
                    continue;
                }
            }
        } else {
            consecutive_health_failures = 0;
            let exit_code = manager.child_pid().and_then(crate::process::try_wait);
            on_event(MonitorEvent::MachineExited { exit_code });
            let _ = service.db.update_vm(&request.name, |r| {
                r.state = RecordState::Stopped;
                r.pid = None;
                r.last_exit_code = exit_code;
            });

            if restart.should_restart(exit_code) {
                let backoff = restart.backoff_duration();
                restart.restart_count += 1;
                on_event(MonitorEvent::Restarting {
                    attempt: restart.restart_count,
                    backoff_secs: backoff.as_secs(),
                });
                let _ = service.db.update_vm(&request.name, |r| {
                    r.restart.restart_count = restart.restart_count;
                });
                std::thread::sleep(backoff);
                if should_stop() {
                    break;
                }
                match service.start(StartMachine::new(request.name.clone())) {
                    Ok(_) => {
                        on_event(MonitorEvent::Restarted);
                        last_start = std::time::Instant::now();
                    }
                    Err(error) => on_event(MonitorEvent::RestartFailed {
                        error: error.to_string(),
                    }),
                }
            } else {
                on_event(MonitorEvent::NotRestarting {
                    policy: restart.policy.clone(),
                    count: restart.restart_count,
                    max_retries: restart.max_retries,
                });
                break;
            }
        }
    }

    let _ = service.db.update_vm(&request.name, |r| {
        r.restart.user_stopped = true;
    });
    on_event(MonitorEvent::Stopped {
        name: request.name.clone(),
    });
    Ok(MonitorResult { name: request.name })
}

/// Request to create a machine.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CreateMachine {
    /// Machine name.
    pub name: String,
    /// OCI image reference.
    pub image: Option<String>,
    /// Entrypoint override.
    pub entrypoint: Vec<String>,
    /// Command override.
    pub cmd: Vec<String>,
    /// vCPU count.
    pub cpus: u8,
    /// Memory in MiB.
    pub memory_mib: u32,
    /// Host directory mounts.
    pub mounts: Vec<HostMount>,
    /// Host-to-guest TCP ports.
    pub ports: Vec<PortMapping>,
    /// Enable networking.
    pub net: bool,
    /// Network backend.
    pub network_backend: Option<NetworkBackend>,
    /// Init commands.
    pub init: Vec<String>,
    /// Env vars in KEY=VALUE form.
    pub env: Vec<String>,
    /// Workdir.
    pub workdir: Option<String>,
    /// User.
    pub user: Option<String>,
    /// Storage GiB.
    pub storage_gb: Option<u64>,
    /// Overlay GiB.
    pub overlay_gb: Option<u64>,
    /// CIDR egress allowlist.
    pub allowed_cidrs: Option<Vec<String>>,
    /// Restart policy.
    pub restart_policy: Option<RestartPolicy>,
    /// Restart retries.
    pub restart_max_retries: Option<u32>,
    /// Restart backoff.
    pub restart_max_backoff_secs: Option<u64>,
    /// Health command.
    pub health_cmd: Option<Vec<String>>,
    /// Health interval seconds.
    pub health_interval_secs: Option<u64>,
    /// Health timeout seconds.
    pub health_timeout_secs: Option<u64>,
    /// Health retries.
    pub health_retries: Option<u32>,
    /// Health startup grace seconds.
    pub health_startup_grace_secs: Option<u64>,
    /// Forward SSH agent.
    pub ssh_agent: bool,
    /// GPU acceleration.
    pub gpu: bool,
    /// GPU VRAM MiB.
    pub gpu_vram_mib: Option<u32>,
    /// DNS filter hosts.
    pub dns_filter_hosts: Option<Vec<String>>,
    /// Source .smolmachine sidecar.
    pub source_smolmachine: Option<String>,
    /// Secret refs.
    pub secret_refs: BTreeMap<String, SecretRef>,
}

impl CreateMachine {
    /// Build a request with default resources.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            image: None,
            entrypoint: Vec::new(),
            cmd: Vec::new(),
            cpus: DEFAULT_MICROVM_CPU_COUNT,
            memory_mib: DEFAULT_MICROVM_MEMORY_MIB,
            mounts: Vec::new(),
            ports: Vec::new(),
            net: false,
            network_backend: None,
            init: Vec::new(),
            env: Vec::new(),
            workdir: None,
            user: None,
            storage_gb: Some(DEFAULT_STORAGE_SIZE_GIB),
            overlay_gb: Some(DEFAULT_OVERLAY_SIZE_GIB),
            allowed_cidrs: None,
            restart_policy: None,
            restart_max_retries: None,
            restart_max_backoff_secs: None,
            health_cmd: None,
            health_interval_secs: None,
            health_timeout_secs: None,
            health_retries: None,
            health_startup_grace_secs: None,
            ssh_agent: false,
            gpu: false,
            gpu_vram_mib: None,
            dns_filter_hosts: None,
            source_smolmachine: None,
            secret_refs: BTreeMap::new(),
        }
    }

    fn into_record(self) -> Result<VmRecord> {
        validate_vm_name(&self.name, "machine name")
            .map_err(|reason| Error::config("create machine", reason))?;
        validate_ports(&self.ports)?;
        validate_requested_network_backend(
            &self.vm_resources(),
            self.dns_filter_hosts.as_deref(),
            self.ports.len(),
        )?;
        for (name, secret_ref) in &self.secret_refs {
            secrets::validate_ref(secret_ref, ResolutionScope::TrustedLocal).map_err(|error| {
                Error::config("create machine", format!("secret '{}': {}", name, error))
            })?;
        }
        let mut record = VmRecord::new_with_restart(
            self.name,
            self.cpus,
            self.memory_mib,
            self.mounts
                .iter()
                .map(HostMount::to_storage_tuple)
                .collect(),
            PortMapping::to_tuples(&self.ports),
            self.net,
            RestartConfig {
                policy: self.restart_policy.unwrap_or(RestartPolicy::Never),
                max_retries: self.restart_max_retries.unwrap_or(0),
                max_backoff_secs: self.restart_max_backoff_secs.unwrap_or(0),
                ..Default::default()
            },
        );
        record.image = self.image;
        record.entrypoint = self.entrypoint;
        record.cmd = self.cmd;
        record.init = self.init;
        record.env = crate::util::parse_env_list(&self.env);
        record.workdir = self.workdir;
        record.user = self.user;
        record.storage_gb = self.storage_gb;
        record.overlay_gb = self.overlay_gb;
        record.allowed_cidrs = self.allowed_cidrs;
        record.network_backend = self.network_backend;
        record.gpu = self.gpu.then_some(true);
        record.gpu_vram_mib = validate_gpu_vram_mib(self.gpu_vram_mib)
            .map_err(|error| Error::config("create machine", format!("gpu_vram: {error}")))?;
        record.health_cmd = self.health_cmd;
        record.health_interval_secs = self.health_interval_secs;
        record.health_timeout_secs = self.health_timeout_secs;
        record.health_retries = self.health_retries;
        record.health_startup_grace_secs = self.health_startup_grace_secs;
        record.ssh_agent = self.ssh_agent;
        record.dns_filter_hosts = self.dns_filter_hosts;
        record.source_smolmachine = self.source_smolmachine;
        record.secret_refs = self.secret_refs;
        Ok(record)
    }

    fn vm_resources(&self) -> crate::data::resources::VmResources {
        crate::data::resources::VmResources {
            cpus: self.cpus,
            memory_mib: self.memory_mib,
            network: self.net,
            allowed_cidrs: self.allowed_cidrs.clone(),
            network_backend: self.network_backend,
            storage_gib: self.storage_gb,
            overlay_gib: self.overlay_gb,
            gpu: self.gpu,
            gpu_vram_mib: self.gpu_vram_mib,
        }
    }
}

/// Request to get one machine.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GetMachine {
    /// Machine name.
    pub name: String,
}

impl GetMachine {
    /// Create a get request.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

/// Request to list machines.
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct ListMachines;

/// Request to start a machine.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StartMachine {
    /// Machine name.
    pub name: String,
    /// Start as forkable golden.
    pub forkable: bool,
    /// Proxy for pulls.
    pub proxy: Option<String>,
    /// No-proxy for pulls.
    pub no_proxy: Option<String>,
    /// Snapshot directory for fork clone boot.
    pub snapshot_dir: Option<PathBuf>,
}

impl StartMachine {
    /// Create a start request.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            forkable: false,
            proxy: None,
            no_proxy: None,
            snapshot_dir: None,
        }
    }
}

/// Request to stop a machine.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StopMachine {
    /// Machine name.
    pub name: String,
}

impl StopMachine {
    /// Create a stop request.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

/// Request to delete a machine.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DeleteMachine {
    /// Machine name.
    pub name: String,
    /// Allow deleting a golden with dependent clones.
    pub break_dependent_clones: bool,
}

impl DeleteMachine {
    /// Create a delete request.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            break_dependent_clones: false,
        }
    }
}

/// Request to fork a machine.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ForkMachine {
    /// Golden machine name.
    pub golden: String,
    /// Clone machine name.
    pub clone: String,
    /// Pinned clone ports; empty remaps from golden automatically.
    pub ports: Vec<PortMapping>,
}

impl ForkMachine {
    /// Create a fork request.
    pub fn new(golden: impl Into<String>, clone: impl Into<String>) -> Self {
        Self {
            golden: golden.into(),
            clone: clone.into(),
            ports: Vec::new(),
        }
    }
}

/// Machine status response.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MachineStatus {
    /// Machine name.
    pub name: String,
    /// Resolved runtime state.
    pub state: RecordState,
    /// Persisted machine record.
    pub record: VmRecord,
}

/// Request to update a stopped machine.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct UpdateMachine {
    /// Machine name.
    pub name: String,
    /// Mounts to add.
    pub add_mounts: Vec<HostMount>,
    /// Mounts to remove, matched by source and target.
    pub remove_mounts: Vec<HostMount>,
    /// Port mappings to add.
    pub add_ports: Vec<PortMapping>,
    /// Port mappings to remove.
    pub remove_ports: Vec<PortMapping>,
    /// New vCPU count.
    pub cpus: Option<u8>,
    /// New memory size in MiB.
    pub memory_mib: Option<u32>,
    /// Enable networking.
    pub enable_network: bool,
    /// Disable networking and clear dependent egress policy.
    pub disable_network: bool,
    /// New network backend.
    pub network_backend: Option<NetworkBackend>,
    /// Replace allowed CIDR policy.
    pub allowed_cidrs: Option<Vec<String>>,
    /// Clear allowed CIDR policy.
    pub clear_allowed_cidrs: bool,
    /// Replace DNS hostname filter policy.
    pub dns_filter_hosts: Option<Vec<String>>,
    /// Clear DNS hostname filter policy.
    pub clear_dns_filter_hosts: bool,
    /// Environment variables to set by key.
    pub set_env: Vec<(String, String)>,
    /// Environment variable keys to remove.
    pub remove_env: Vec<String>,
    /// Replace the configured workdir.
    pub workdir: Option<String>,
    /// Clear the configured workdir.
    pub clear_workdir: bool,
    /// Enable GPU acceleration.
    pub enable_gpu: bool,
    /// Disable GPU acceleration.
    pub disable_gpu: bool,
    /// GPU VRAM size in MiB.
    pub gpu_vram_mib: Option<u32>,
    /// Expand storage disk to this size in GiB.
    pub storage_gb: Option<u64>,
    /// Expand overlay disk to this size in GiB.
    pub overlay_gb: Option<u64>,
    /// Replace restart policy.
    pub restart_policy: Option<RestartPolicy>,
    /// Replace max restart retries.
    pub restart_max_retries: Option<u32>,
    /// Replace max restart backoff seconds.
    pub restart_max_backoff_secs: Option<u64>,
    /// Replace health check command.
    pub health_cmd: Option<Vec<String>>,
    /// Clear health check command.
    pub clear_health_cmd: bool,
    /// Replace health interval seconds.
    pub health_interval_secs: Option<u64>,
    /// Replace health timeout seconds.
    pub health_timeout_secs: Option<u64>,
    /// Replace health retries.
    pub health_retries: Option<u32>,
    /// Replace health startup grace seconds.
    pub health_startup_grace_secs: Option<u64>,
    /// Enable SSH agent forwarding for future starts.
    pub enable_ssh_agent: bool,
    /// Disable SSH agent forwarding for future starts.
    pub disable_ssh_agent: bool,
}

impl UpdateMachine {
    /// Build an empty update request.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            add_mounts: Vec::new(),
            remove_mounts: Vec::new(),
            add_ports: Vec::new(),
            remove_ports: Vec::new(),
            cpus: None,
            memory_mib: None,
            enable_network: false,
            disable_network: false,
            network_backend: None,
            allowed_cidrs: None,
            clear_allowed_cidrs: false,
            dns_filter_hosts: None,
            clear_dns_filter_hosts: false,
            set_env: Vec::new(),
            remove_env: Vec::new(),
            workdir: None,
            clear_workdir: false,
            enable_gpu: false,
            disable_gpu: false,
            gpu_vram_mib: None,
            storage_gb: None,
            overlay_gb: None,
            restart_policy: None,
            restart_max_retries: None,
            restart_max_backoff_secs: None,
            health_cmd: None,
            clear_health_cmd: false,
            health_interval_secs: None,
            health_timeout_secs: None,
            health_retries: None,
            health_startup_grace_secs: None,
            enable_ssh_agent: false,
            disable_ssh_agent: false,
        }
    }
}

/// Result from updating a machine.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct UpdateResult {
    /// Updated status.
    pub status: MachineStatus,
    /// Human-readable change summary.
    pub changes: Vec<String>,
}

/// Request to execute a command in a machine.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ExecMachine {
    /// Machine name.
    pub name: String,
    /// Command argv.
    pub command: Vec<String>,
    /// Environment variables to layer on top of selected defaults.
    pub env: Vec<(String, String)>,
    /// Ad-hoc secret references for this exec.
    pub secret_refs: BTreeMap<String, SecretRef>,
    /// Trust scope used to resolve ad-hoc secrets.
    pub secret_scope: ResolutionScope,
    /// Working directory override.
    pub workdir: Option<String>,
    /// Execution timeout.
    pub timeout: Option<Duration>,
    /// Stdin text for buffered bare-VM exec.
    pub stdin: Option<String>,
    /// Run in background and return a PID.
    pub background: bool,
    /// Allocate a TTY for interactive execution.
    pub tty: bool,
    /// Start the machine if it is stopped.
    pub start_if_needed: bool,
    /// Include persisted record env and secret refs.
    pub include_record_env: bool,
    /// Trace identifier propagated to the agent.
    pub trace_id: Option<String>,
}

impl ExecMachine {
    /// Build an exec request.
    pub fn new(name: impl Into<String>, command: Vec<String>) -> Self {
        Self {
            name: name.into(),
            command,
            env: Vec::new(),
            secret_refs: BTreeMap::new(),
            secret_scope: ResolutionScope::TrustedLocal,
            workdir: None,
            timeout: None,
            stdin: None,
            background: false,
            tty: false,
            start_if_needed: false,
            include_record_env: true,
            trace_id: None,
        }
    }
}

/// Request to run an image command in an existing machine.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RunMachine {
    /// Machine name.
    pub name: String,
    /// OCI image reference.
    pub image: String,
    /// Command argv.
    pub command: Vec<String>,
    /// Environment variables.
    pub env: Vec<(String, String)>,
    /// Ad-hoc secret references.
    pub secret_refs: BTreeMap<String, SecretRef>,
    /// Trust scope used to resolve ad-hoc secrets.
    pub secret_scope: ResolutionScope,
    /// Working directory.
    pub workdir: Option<String>,
    /// Execution timeout.
    pub timeout: Option<Duration>,
    /// Persistent overlay id.
    pub persistent_overlay_id: Option<String>,
    /// Start the machine if stopped.
    pub start_if_needed: bool,
    /// Trace identifier propagated to the agent.
    pub trace_id: Option<String>,
}

impl RunMachine {
    /// Build a run request.
    pub fn new(name: impl Into<String>, image: impl Into<String>, command: Vec<String>) -> Self {
        Self {
            name: name.into(),
            image: image.into(),
            command,
            env: Vec::new(),
            secret_refs: BTreeMap::new(),
            secret_scope: ResolutionScope::TrustedLocal,
            workdir: None,
            timeout: None,
            persistent_overlay_id: None,
            start_if_needed: false,
            trace_id: None,
        }
    }
}

/// I/O callbacks for a local `machine run` style session.
pub trait MachineRunIo {
    /// Called before the VM is started.
    fn starting(&mut self, _event: &MachineRunStarting) {}
    /// Called as an image pull reports progress.
    fn pull_progress(&mut self, _event: &MachineRunPullProgress) {}
    /// Called with buffered stdout from a non-interactive foreground command.
    fn stdout(&mut self, _bytes: &[u8]) {}
    /// Called with buffered stderr from a non-interactive foreground command.
    fn stderr(&mut self, _bytes: &[u8]) {}
}

impl MachineRunIo for () {}

/// Start notification for a `machine run` style session.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct MachineRunStarting {
    /// Runtime machine name.
    pub name: String,
    /// Whether the session will persist and detach after launch.
    pub detached: bool,
}

/// Image pull progress from a `machine run` style session.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct MachineRunPullProgress {
    /// Image being pulled.
    pub image: String,
    /// Current progress unit.
    pub current: usize,
    /// Total progress unit.
    pub total: usize,
    /// Current layer identifier, or the sentinel `syncing`.
    pub layer: String,
}

/// Request for a foreground or detached `machine run` style session.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MachineRun {
    /// Runtime machine name.
    pub name: String,
    /// Persist the machine record and leave the VM running.
    pub detached: bool,
    /// Allow an existing record with the same name to be updated.
    pub allow_existing_record: bool,
    /// OCI image reference. `None` runs directly in the bare VM.
    pub image: Option<String>,
    /// Explicit command argv from the adapter.
    pub command: Vec<String>,
    /// Entrypoint from declarative configuration.
    pub entrypoint: Vec<String>,
    /// Command from declarative configuration.
    pub cmd: Vec<String>,
    /// Non-secret environment variables.
    pub env: Vec<(String, String)>,
    /// Ad-hoc secret references for this launch.
    pub secret_refs: BTreeMap<String, SecretRef>,
    /// Trust scope used to resolve ad-hoc secrets.
    pub secret_scope: ResolutionScope,
    /// Working directory.
    pub workdir: Option<String>,
    /// Container user.
    pub user: Option<String>,
    /// Host directory mounts.
    pub mounts: Vec<HostMount>,
    /// Host-to-guest TCP port mappings.
    pub ports: Vec<PortMapping>,
    /// VM resources.
    pub resources: VmResources,
    /// Init commands to run after first boot.
    pub init: Vec<String>,
    /// Forward the host SSH agent.
    pub ssh_agent: bool,
    /// Hostnames used for DNS filtering.
    pub dns_filter_hosts: Option<Vec<String>>,
    /// Target OCI platform for image pulls.
    pub oci_platform: Option<String>,
    /// Proxy URL for image pulls.
    pub proxy: Option<String>,
    /// No-proxy list for image pulls.
    pub no_proxy: Option<String>,
    /// Execution timeout.
    pub timeout: Option<Duration>,
    /// Keep stdin open and use the process terminal.
    pub interactive: bool,
    /// Allocate a pseudo-TTY.
    pub tty: bool,
    /// Install a local SIGINT guard that kills the VM during setup/non-interactive execution.
    pub kill_on_sigint: bool,
}

impl MachineRun {
    /// Build a foreground bare-VM run request.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            detached: false,
            allow_existing_record: false,
            image: None,
            command: Vec::new(),
            entrypoint: Vec::new(),
            cmd: Vec::new(),
            env: Vec::new(),
            secret_refs: BTreeMap::new(),
            secret_scope: ResolutionScope::TrustedLocal,
            workdir: None,
            user: None,
            mounts: Vec::new(),
            ports: Vec::new(),
            resources: VmResources::default(),
            init: Vec::new(),
            ssh_agent: false,
            dns_filter_hosts: None,
            oci_platform: None,
            proxy: None,
            no_proxy: None,
            timeout: None,
            interactive: false,
            tty: false,
            kill_on_sigint: false,
        }
    }
}

/// Result from a `machine run` style session.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MachineRunResult {
    /// A foreground command completed.
    Foreground {
        /// Process exit code.
        exit_code: i32,
        /// Captured stdout for non-interactive commands.
        stdout: Vec<u8>,
        /// Captured stderr for non-interactive commands.
        stderr: Vec<u8>,
    },
    /// A detached machine was persisted and left running.
    Detached {
        /// Machine name.
        name: String,
        /// VMM process id when known.
        pid: Option<i32>,
    },
}

/// Request to monitor a machine's restart policy and health check.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MonitorMachine {
    /// Machine name.
    pub name: String,
    /// Restart policy override.
    pub restart_policy: Option<RestartPolicy>,
    /// Health command override.
    pub health_cmd: Option<Vec<String>>,
    /// Default health timeout when the record omits one.
    pub health_timeout: Duration,
    /// Default check interval when the record omits one.
    pub interval: Duration,
    /// Default health retry count when the record omits one.
    pub health_retries: u32,
}

impl MonitorMachine {
    /// Build a monitor request.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            restart_policy: None,
            health_cmd: None,
            health_timeout: Duration::from_secs(5),
            interval: Duration::from_secs(5),
            health_retries: 3,
        }
    }
}

/// Effective health configuration used by a monitor session.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct MonitorHealthConfig {
    /// Health command argv.
    pub command: Vec<String>,
    /// Timeout seconds.
    pub timeout_secs: u64,
    /// Retry count.
    pub retries: u32,
}

/// Events emitted by [`MachineService::monitor`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MonitorEvent {
    /// The machine was stopped when monitoring began and is being started.
    Starting {
        /// Machine name.
        name: String,
    },
    /// Monitoring has begun.
    Monitoring {
        /// Machine name.
        name: String,
        /// Effective restart policy.
        policy: RestartPolicy,
        /// Check interval seconds.
        interval_secs: u64,
        /// Effective health config, if any.
        health: Option<MonitorHealthConfig>,
    },
    /// A long sleep/suspend was detected.
    SuspendDetected {
        /// Extra sleep duration seconds beyond the configured interval.
        sleep_secs: u64,
    },
    /// A previously failing health check recovered.
    HealthRecovered,
    /// A health command exited non-zero.
    HealthFailed {
        /// Exit code.
        exit_code: i32,
        /// Consecutive failure count.
        consecutive: u32,
        /// Retry threshold.
        retries: u32,
        /// Stderr text.
        stderr: String,
    },
    /// A health command could not be executed.
    HealthError {
        /// Consecutive failure count.
        consecutive: u32,
        /// Retry threshold.
        retries: u32,
        /// Error text.
        error: String,
    },
    /// The agent could not be reached for a health check.
    AgentUnreachable {
        /// Consecutive failure count.
        consecutive: u32,
        /// Retry threshold.
        retries: u32,
    },
    /// Health retry threshold was reached; the VM is being stopped for restart.
    UnhealthyStopping,
    /// The VMM process exited.
    MachineExited {
        /// Exit code when known.
        exit_code: Option<i32>,
    },
    /// The monitor is sleeping before restarting the machine.
    Restarting {
        /// Restart attempt number.
        attempt: u32,
        /// Backoff seconds.
        backoff_secs: u64,
    },
    /// The machine restarted successfully.
    Restarted,
    /// A restart failed.
    RestartFailed {
        /// Error text.
        error: String,
    },
    /// Restart policy declined another restart.
    NotRestarting {
        /// Effective restart policy.
        policy: RestartPolicy,
        /// Restart count.
        count: u32,
        /// Max retries; `0` means unlimited.
        max_retries: u32,
    },
    /// Monitoring stopped.
    Stopped {
        /// Machine name.
        name: String,
    },
}

/// Result from a monitor session.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct MonitorResult {
    /// Machine name.
    pub name: String,
}

/// Buffered exec result.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ExecResult {
    /// Process exit code.
    pub exit_code: i32,
    /// Raw stdout bytes.
    pub stdout: Vec<u8>,
    /// Raw stderr bytes.
    pub stderr: Vec<u8>,
}

/// Request to write bytes to a machine file.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct WriteMachineFile {
    /// Machine name.
    pub name: String,
    /// Absolute guest path.
    pub guest_path: String,
    /// Bytes to write.
    pub data: Vec<u8>,
    /// Optional file mode.
    pub mode: Option<u32>,
    /// Start the machine if stopped.
    pub start_if_needed: bool,
    /// Trace identifier propagated to the agent.
    pub trace_id: Option<String>,
}

/// Request to upload a host file to a machine.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct UploadMachineFile {
    /// Machine name.
    pub name: String,
    /// Host path to read from.
    pub local_path: PathBuf,
    /// Absolute guest path to write.
    pub guest_path: String,
    /// Optional file mode.
    pub mode: Option<u32>,
    /// Start the machine if stopped.
    pub start_if_needed: bool,
    /// Trace identifier propagated to the agent.
    pub trace_id: Option<String>,
}

/// Request to read a machine file into memory.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ReadMachineFile {
    /// Machine name.
    pub name: String,
    /// Absolute guest path.
    pub guest_path: String,
    /// Start the machine if stopped.
    pub start_if_needed: bool,
    /// Trace identifier propagated to the agent.
    pub trace_id: Option<String>,
}

/// Request to download a machine file to a host path.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DownloadMachineFile {
    /// Machine name.
    pub name: String,
    /// Absolute guest path to read.
    pub guest_path: String,
    /// Host path to write.
    pub local_path: PathBuf,
    /// Start the machine if stopped.
    pub start_if_needed: bool,
    /// Trace identifier propagated to the agent.
    pub trace_id: Option<String>,
}

/// File transfer result.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct FileTransfer {
    /// Bytes transferred.
    pub bytes: u64,
}

/// Request for machine storage status.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StorageStatusRequest {
    /// Machine name.
    pub name: String,
    /// Start the machine if stopped.
    pub start_if_needed: bool,
    /// Trace identifier propagated to the agent.
    pub trace_id: Option<String>,
}

/// Request to list cached images.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ListMachineImages {
    /// Machine name.
    pub name: String,
    /// Start the machine if stopped.
    pub start_if_needed: bool,
    /// Stop the machine after this operation if this service started it.
    pub stop_after_start: bool,
    /// Return an empty list instead of an error when stopped.
    pub empty_when_stopped: bool,
    /// Trace identifier propagated to the agent.
    pub trace_id: Option<String>,
}

/// Request to pull an image into a machine.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PullMachineImage {
    /// Machine name.
    pub name: String,
    /// OCI image reference.
    pub image: String,
    /// Target OCI platform.
    pub oci_platform: Option<String>,
    /// Proxy URL.
    pub proxy: Option<String>,
    /// No-proxy list.
    pub no_proxy: Option<String>,
    /// Start the machine if stopped.
    pub start_if_needed: bool,
    /// Trace identifier propagated to the agent.
    pub trace_id: Option<String>,
}

/// Request to prune machine image storage.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PruneMachineImages {
    /// Machine name.
    pub name: String,
    /// Dry-run only.
    pub dry_run: bool,
    /// Remove all cached manifests/layers.
    pub all: bool,
    /// Stop the machine after this operation if this service started it.
    pub stop_after_start: bool,
    /// Trace identifier propagated to the agent.
    pub trace_id: Option<String>,
}

/// Result from pruning image storage.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct PruneImagesResult {
    /// Bytes freed, or bytes that would be freed for a dry run.
    pub freed_bytes: u64,
    /// Count of removed images when known.
    pub removed_images: usize,
    /// Whether this was a dry run.
    pub dry_run: bool,
}

/// Request to run a network diagnostic.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct NetworkTestMachine {
    /// Machine name.
    pub name: String,
    /// URL to test.
    pub url: String,
    /// Start the machine if stopped.
    pub start_if_needed: bool,
    /// Trace identifier propagated to the agent.
    pub trace_id: Option<String>,
}

/// Request for a machine data directory.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DataDirMachine {
    /// Machine name.
    pub name: String,
    /// Require the machine to exist in the database.
    pub require_existing: bool,
}

impl WriteMachineFile {
    /// Build a write-file request.
    pub fn new(name: impl Into<String>, guest_path: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            guest_path: guest_path.into(),
            data,
            mode: None,
            start_if_needed: false,
            trace_id: None,
        }
    }
}

impl UploadMachineFile {
    /// Build an upload request.
    pub fn new(
        name: impl Into<String>,
        local_path: impl Into<PathBuf>,
        guest_path: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            local_path: local_path.into(),
            guest_path: guest_path.into(),
            mode: None,
            start_if_needed: false,
            trace_id: None,
        }
    }
}

impl ReadMachineFile {
    /// Build a read-file request.
    pub fn new(name: impl Into<String>, guest_path: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            guest_path: guest_path.into(),
            start_if_needed: false,
            trace_id: None,
        }
    }
}

impl DownloadMachineFile {
    /// Build a download request.
    pub fn new(
        name: impl Into<String>,
        guest_path: impl Into<String>,
        local_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            name: name.into(),
            guest_path: guest_path.into(),
            local_path: local_path.into(),
            start_if_needed: false,
            trace_id: None,
        }
    }
}

impl StorageStatusRequest {
    /// Build a storage-status request.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            start_if_needed: false,
            trace_id: None,
        }
    }
}

impl ListMachineImages {
    /// Build a list-images request.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            start_if_needed: false,
            stop_after_start: false,
            empty_when_stopped: false,
            trace_id: None,
        }
    }
}

impl PullMachineImage {
    /// Build a pull-image request.
    pub fn new(name: impl Into<String>, image: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            image: image.into(),
            oci_platform: None,
            proxy: None,
            no_proxy: None,
            start_if_needed: false,
            trace_id: None,
        }
    }
}

impl PruneMachineImages {
    /// Build a prune-images request.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            dry_run: false,
            all: false,
            stop_after_start: false,
            trace_id: None,
        }
    }
}

impl NetworkTestMachine {
    /// Build a network-test request.
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            url: url.into(),
            start_if_needed: false,
            trace_id: None,
        }
    }
}

impl DataDirMachine {
    /// Build a data-dir request.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            require_existing: true,
        }
    }
}

fn prepare_packed_layers_for_create(record: &VmRecord) -> Result<()> {
    let Some(sidecar_path) = record.source_smolmachine.as_ref() else {
        return Ok(());
    };
    let sidecar = PathBuf::from(sidecar_path);
    if !sidecar.exists() {
        return Err(Error::config(
            "create machine",
            format!("sidecar file not found: {}", sidecar.display()),
        ));
    }

    let _manager =
        AgentManager::for_vm_with_sizes(&record.name, record.storage_gb, record.overlay_gb)
            .map_err(|error| Error::agent("create agent manager", error.to_string()))?;
    let cache_dir = machine_layers_cache_dir(&record.name);
    smolvm_pack::extract::force_detach_layers_volume(&cache_dir);
    match std::fs::remove_dir_all(&cache_dir) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(Error::agent("clear packed layers cache", error.to_string())),
    }

    let result = (|| -> Result<()> {
        let footer = smolvm_pack::packer::read_footer_from_sidecar(&sidecar)
            .map_err(|error| Error::agent("read sidecar footer", error.to_string()))?;
        smolvm_pack::extract::extract_sidecar(&sidecar, &cache_dir, &footer, false, false)
            .map_err(|error| Error::agent("extract sidecar", error.to_string()))
    })();
    smolvm_pack::extract::force_detach_layers_volume(&cache_dir);
    result
}

fn rollback_created_data_dir(name: &str) {
    let cache_dir = machine_layers_cache_dir(name);
    smolvm_pack::extract::force_detach_layers_volume(&cache_dir);
    let data_dir = vm_data_dir(name);
    if let Err(error) = std::fs::remove_dir_all(&data_dir) {
        if error.kind() != std::io::ErrorKind::NotFound {
            tracing::warn!(machine = %name, dir = %data_dir.display(), error = %error, "failed to remove machine data dir after create failure");
        }
    }
}

struct CreateReservation<'a> {
    db: &'a SmolvmDb,
    name: String,
    token: String,
    completed: bool,
}

impl<'a> CreateReservation<'a> {
    fn reserve(db: &'a SmolvmDb, name: &str) -> Result<Self> {
        let token = SmolvmDb::create_reservation_token();
        if !db.reserve_vm_create(name, &token)? {
            return Err(Error::config(
                "create machine",
                format!("machine '{name}' already exists or is being created"),
            ));
        }
        Ok(Self {
            db,
            name: name.to_string(),
            token,
            completed: false,
        })
    }

    fn commit(mut self, record: &VmRecord) -> Result<()> {
        if !self
            .db
            .commit_reserved_vm(&self.name, &self.token, record)?
        {
            return Err(Error::config(
                "create machine",
                format!(
                    "machine '{}' already exists or is no longer reserved",
                    self.name
                ),
            ));
        }
        self.completed = true;
        Ok(())
    }
}

impl Drop for CreateReservation<'_> {
    fn drop(&mut self) {
        if !self.completed {
            let _ = self
                .db
                .release_vm_create_reservation(&self.name, &self.token);
        }
    }
}

fn status_from_record(name: &str, record: VmRecord) -> MachineStatus {
    MachineStatus {
        name: name.to_string(),
        state: crate::agent::state_probe::resolve_state(name, &record),
        record,
    }
}

fn get_record(db: &SmolvmDb, name: &str) -> Result<VmRecord> {
    db.get_vm(name)?.ok_or_else(|| Error::vm_not_found(name))
}

fn mark_running(db: &SmolvmDb, name: &str, pid: Option<i32>) -> Result<()> {
    let pid_start_time = pid.and_then(crate::process::process_start_time);
    db.update_vm(name, |r| {
        r.state = RecordState::Running;
        r.pid = pid;
        r.pid_start_time = pid_start_time;
    })?
    .ok_or_else(|| Error::vm_not_found(name))?;
    Ok(())
}

fn mark_stopped(db: &SmolvmDb, name: &str) -> Result<()> {
    db.update_vm(name, |r| {
        r.state = RecordState::Stopped;
        r.pid = None;
        r.pid_start_time = None;
    })?
    .ok_or_else(|| Error::vm_not_found(name))?;
    Ok(())
}

fn launch_features(request: &StartMachine, record: &VmRecord) -> Result<LaunchFeatures> {
    let ssh_agent_socket = if record.ssh_agent {
        Some(
            std::env::var_os("SSH_AUTH_SOCK")
                .ok_or_else(|| Error::config("ssh-agent", "SSH_AUTH_SOCK is not set"))?
                .into(),
        )
    } else {
        None
    };

    LaunchFeatures {
        ssh_agent_socket,
        dns_filter_hosts: record.dns_filter_hosts.clone(),
        control_socket: request.forkable.then(|| control_socket_path(&request.name)),
        snapshot_dir: request.snapshot_dir.clone(),
        ..Default::default()
    }
    .with_packed_layers(
        &machine_layers_cache_dir(&request.name),
        record.source_smolmachine.as_deref(),
    )
}

fn record_env_with_secrets(record: &VmRecord) -> Result<Vec<(String, String)>> {
    let mut env = record.env.clone();
    env.extend(secrets::expose_into_env(secrets::resolve_refs_to_env(
        &record.secret_refs,
        ResolutionScope::RecordReplay,
    )?));
    Ok(env)
}

fn run_first_start(
    db: &SmolvmDb,
    name: &str,
    record: &mut VmRecord,
    client: &mut AgentClient,
    env: &[(String, String)],
    request: &StartMachine,
) -> Result<()> {
    if let Some(image) = record.image.as_ref() {
        if record.source_smolmachine.is_none() {
            let mut opts = PullOptions::new().use_registry_config(true);
            if let Some(proxy) = request.proxy.clone() {
                opts = opts.proxy(proxy);
            }
            if let Some(no_proxy) = request.no_proxy.clone() {
                opts = opts.no_proxy(no_proxy);
            }
            let info = client.pull(image, opts)?;
            if record.entrypoint.is_empty() && record.cmd.is_empty() {
                record.entrypoint = info.entrypoint;
                record.cmd = info.cmd;
                let ep = record.entrypoint.clone();
                let cmd = record.cmd.clone();
                let _ = db.update_vm(name, |r| {
                    r.entrypoint = ep;
                    r.cmd = cmd;
                });
            }
        }
    }

    for (i, cmd) in record.init.iter().enumerate() {
        let (code, stdout, stderr) = if let Some(image) = record.image.as_ref() {
            let config = RunConfig::new(image, vec!["/bin/sh".into(), "-c".into(), cmd.clone()])
                .with_env(env.to_vec())
                .with_workdir(record.workdir.clone())
                .with_user(record.user.clone())
                .with_mounts(run_mounts(&record.mounts))
                .with_persistent_overlay(Some(name.to_string()));
            client.run_non_interactive(config)?
        } else {
            client.vm_exec(
                vec!["/bin/sh".into(), "-c".into(), cmd.clone()],
                env.to_vec(),
                record.workdir.clone(),
                None,
                None,
            )?
        };
        if code != 0 {
            return Err(Error::agent(
                "run init",
                format!(
                    "init[{i}] failed (exit {code}): {}; stdout: {}",
                    String::from_utf8_lossy(&stderr).trim(),
                    String::from_utf8_lossy(&stdout).trim()
                ),
            ));
        }
    }
    Ok(())
}

fn launch_workload(
    name: &str,
    record: &VmRecord,
    client: &mut AgentClient,
    env: Vec<(String, String)>,
) -> Result<()> {
    let Some(image) = record.image.as_ref() else {
        return Ok(());
    };
    let mut cmd = record.entrypoint.clone();
    cmd.extend(record.cmd.clone());
    let config = RunConfig::new(image, cmd)
        .with_env(env)
        .with_workdir(record.workdir.clone())
        .with_user(record.user.clone())
        .with_mounts(run_mounts(&record.mounts))
        .with_persistent_overlay(Some(name.to_string()));
    client.run_container_detached(config).map(|_| ())
}

fn run_mounts(mounts: &[(String, String, bool)]) -> Vec<(String, String, bool)> {
    mounts
        .iter()
        .enumerate()
        .map(|(i, (_, target, ro))| (HostMount::mount_tag(i), target.clone(), *ro))
        .collect()
}

fn validate_ports(ports: &[PortMapping]) -> Result<()> {
    PortMapping::check_duplicates(ports).map_err(|e| Error::config("create machine", e))?;
    for p in ports {
        if p.host == 0 || p.guest == 0 {
            return Err(Error::config(
                "create machine",
                "port 0 is not valid for VM port forwarding",
            ));
        }
    }
    Ok(())
}

fn control_socket_path(name: &str) -> PathBuf {
    vm_data_dir(name).join("control.sock")
}

fn control_socket_cmd(sock: &std::path::Path, cmd: &str) -> Result<String> {
    use std::io::Read;
    use std::os::unix::net::UnixStream;

    let mut stream = UnixStream::connect(sock)
        .map_err(|e| Error::agent("connect control socket", e.to_string()))?;
    stream
        .write_all(format!("{cmd}\n").as_bytes())
        .map_err(|e| Error::agent("write control socket", e.to_string()))?;
    let mut reply = String::new();
    stream
        .read_to_string(&mut reply)
        .map_err(|e| Error::agent("read control socket", e.to_string()))?;
    Ok(reply.lines().next().unwrap_or_default().to_string())
}

fn clone_ports(golden: &VmRecord, pinned: &[PortMapping]) -> Vec<(u16, u16)> {
    if !pinned.is_empty() {
        return PortMapping::to_tuples(pinned);
    }
    golden
        .ports
        .iter()
        .filter_map(|(_, guest)| {
            std::net::TcpListener::bind(("127.0.0.1", 0))
                .ok()
                .and_then(|l| l.local_addr().ok())
                .map(|addr| (addr.port(), *guest))
        })
        .collect()
}

fn clone_disks(golden: &str, clone: &str) -> Result<()> {
    let gdir = vm_data_dir(golden);
    let cdir = vm_data_dir(clone);
    std::fs::create_dir_all(&cdir).map_err(|e| Error::agent("create clone dir", e.to_string()))?;
    let disks = [
        crate::storage::STORAGE_DISK_FILENAME,
        crate::storage::OVERLAY_DISK_FILENAME,
    ];

    #[cfg(target_os = "linux")]
    {
        let mut specs = Vec::new();
        for raw in disks {
            let (src, fmt) = resolve_disk_image(&gdir, raw);
            if src.exists() {
                specs.push((
                    cdir.join(std::path::Path::new(raw).with_extension("qcow2")),
                    src.canonicalize()
                        .map_err(|e| Error::agent("clone disk", e.to_string()))?,
                    fmt,
                ));
            }
        }
        crate::agent::create_disk_overlays(&specs)?;
    }

    #[cfg(target_os = "macos")]
    {
        for raw in disks {
            let (src, _) = resolve_disk_image(&gdir, raw);
            if src.exists() {
                crate::disk_utils::clone_or_copy_file(&src, &cdir.join(src.file_name().unwrap()))?;
            }
        }
    }

    for raw in disks {
        let marker = std::path::Path::new(raw).with_extension("formatted");
        let src = gdir.join(&marker);
        if src.exists() {
            let _ = std::fs::copy(src, cdir.join(marker));
        }
    }
    Ok(())
}

fn rejuvenate_clone(clone: &str) {
    let Ok(manager) = AgentManager::for_vm(clone) else {
        return;
    };
    if manager.try_connect_existing().is_none() {
        return;
    }
    let Ok(mut client) = AgentClient::connect_with_retry(manager.vsock_socket()) else {
        return;
    };
    let script = format!(
        "hostname '{}' 2>/dev/null || true\nrm -f /etc/machine-id\ndbus-uuidgen --ensure=/etc/machine-id 2>/dev/null || true\n",
        clone
    );
    let _ = client.vm_exec(
        vec!["/bin/sh".into(), "-c".into(), script],
        vec![],
        None,
        Some(std::time::Duration::from_secs(10)),
        None,
    );
    manager.detach();
}

struct PreparedExec {
    record: VmRecord,
    client: AgentClient,
    env: Vec<(String, String)>,
    workdir: Option<String>,
    timeout: Option<Duration>,
    tty: bool,
}

impl PreparedExec {
    fn image_run_config(&mut self, image: String, command: Vec<String>) -> Result<RunConfig> {
        let image_info = ensure_image_present(&mut self.client, &image)?;
        let defaults = resolve_image_runtime_defaults(
            image_info.as_ref(),
            &self.env,
            self.workdir.as_deref(),
            self.record.user.as_deref(),
        );
        Ok(RunConfig::new(image, command)
            .with_env(defaults.env)
            .with_workdir(defaults.workdir)
            .with_user(defaults.user)
            .with_mounts(record_mounts_to_runconfig_bindings(&self.record.mounts))
            .with_timeout(self.timeout)
            .with_tty(self.tty)
            .with_persistent_overlay(Some(self.record.name.clone())))
    }
}

struct ImageRuntimeDefaults {
    env: Vec<(String, String)>,
    workdir: Option<String>,
    user: Option<String>,
}

fn resolve_image_runtime_defaults(
    image_info: Option<&ImageInfo>,
    env: &[(String, String)],
    explicit_workdir: Option<&str>,
    explicit_user: Option<&str>,
) -> ImageRuntimeDefaults {
    let mut resolved_env = Vec::new();
    if let Some(image_info) = image_info {
        for spec in &image_info.env {
            if let Some((key, value)) = crate::util::parse_env_spec(spec) {
                apply_env_override(&mut resolved_env, key, value);
            }
        }
    }
    resolved_env = merge_env_overrides(&resolved_env, env);
    let workdir = explicit_workdir
        .map(str::to_string)
        .or_else(|| image_info.and_then(|info| info.workdir.clone()));
    let user = explicit_user
        .map(str::to_string)
        .or_else(|| image_info.and_then(|info| info.user.clone()));
    ImageRuntimeDefaults {
        env: resolved_env,
        workdir,
        user,
    }
}

fn merge_env_overrides(
    base_env: &[(String, String)],
    overrides: &[(String, String)],
) -> Vec<(String, String)> {
    let mut env = base_env.to_vec();
    for (key, value) in overrides {
        apply_env_override(&mut env, key.clone(), value.clone());
    }
    env
}

fn apply_env_override(env: &mut Vec<(String, String)>, key: String, value: String) {
    env.retain(|(existing, _)| existing != &key);
    env.push((key, value));
}

fn request_env(
    env: &[(String, String)],
    secret_refs: &BTreeMap<String, SecretRef>,
    secret_scope: ResolutionScope,
) -> Result<Vec<(String, String)>> {
    for (name, secret_ref) in secret_refs {
        secrets::validate_ref(secret_ref, secret_scope)
            .map_err(|error| Error::config("secrets", format!("secret '{name}': {error}")))?;
    }
    let mut merged = Vec::new();
    merged.extend(secrets::expose_into_env(secrets::resolve_refs_to_env(
        secret_refs,
        secret_scope,
    )?));
    Ok(merge_env_overrides(&merged, env))
}

fn ensure_image_present(client: &mut AgentClient, image: &str) -> Result<Option<ImageInfo>> {
    match client.query(image)? {
        Some(info) => Ok(Some(info)),
        None => client.pull_with_registry_config(image).map(Some),
    }
}

fn record_mounts_to_runconfig_bindings(
    mounts: &[(String, String, bool)],
) -> Vec<(String, String, bool)> {
    mounts
        .iter()
        .enumerate()
        .map(|(i, (_, target, readonly))| (HostMount::mount_tag(i), target.clone(), *readonly))
        .collect()
}

fn prepare_image_overlay_for_file_ops(
    record: &VmRecord,
    client: &mut AgentClient,
    machine_name: &str,
) -> Result<()> {
    if let Some(image) = record.image.as_ref() {
        ensure_image_present(client, image)?;
        client.run_non_interactive(
            RunConfig::new(image.clone(), vec!["/bin/true".to_string()])
                .with_persistent_overlay(Some(machine_name.to_string())),
        )?;
    }
    Ok(())
}

fn proposed_ports(record: &VmRecord, request: &UpdateMachine) -> Result<Vec<PortMapping>> {
    let mut final_ports: Vec<PortMapping> = record
        .ports
        .iter()
        .filter(|&&(host, guest)| {
            !request
                .remove_ports
                .iter()
                .any(|port| port.host == host && port.guest == guest)
        })
        .map(|&(host, guest)| PortMapping::new(host, guest))
        .collect();
    for port in &request.add_ports {
        if !final_ports
            .iter()
            .any(|existing| existing.host == port.host && existing.guest == port.guest)
        {
            final_ports.push(*port);
        }
    }
    validate_ports(&final_ports)?;
    Ok(final_ports)
}

fn expand_machine_disks(
    name: &str,
    record: &VmRecord,
    storage_gb: Option<u64>,
    overlay_gb: Option<u64>,
) -> Result<Vec<String>> {
    let current_storage_gb = record.storage_gb.unwrap_or(DEFAULT_STORAGE_SIZE_GIB);
    let current_overlay_gb = record.overlay_gb.unwrap_or(DEFAULT_OVERLAY_SIZE_GIB);
    if storage_gb.unwrap_or(current_storage_gb) < current_storage_gb {
        return Err(Error::config(
            "update",
            format!("storage cannot be smaller than current size ({current_storage_gb} GiB)"),
        ));
    }
    if overlay_gb.unwrap_or(current_overlay_gb) < current_overlay_gb {
        return Err(Error::config(
            "update",
            format!("overlay cannot be smaller than current size ({current_overlay_gb} GiB)"),
        ));
    }

    let manager = AgentManager::for_vm(name)
        .map_err(|error| Error::agent("create agent manager", error.to_string()))?;
    let mut changes = Vec::new();
    if let Some(size) = storage_gb.filter(|size| *size > current_storage_gb) {
        expand_disk::<crate::data::disk::Storage>(manager.storage_path(), size)?;
        changes.push(format!("storage: {current_storage_gb} GiB → {size} GiB"));
    }
    if let Some(size) = overlay_gb.filter(|size| *size > current_overlay_gb) {
        expand_disk::<crate::data::disk::Overlay>(manager.overlay_path(), size)?;
        changes.push(format!("overlay: {current_overlay_gb} GiB → {size} GiB"));
    }
    Ok(changes)
}

fn apply_update(
    record: &mut VmRecord,
    request: &UpdateMachine,
    gpu_vram_mib: Option<u32>,
    changes: &mut Vec<String>,
) {
    if let Some(storage_gb) = request.storage_gb {
        record.storage_gb = Some(storage_gb);
    }
    if let Some(overlay_gb) = request.overlay_gb {
        record.overlay_gb = Some(overlay_gb);
    }

    for remove in &request.remove_mounts {
        let remove_tuple = remove.to_storage_tuple();
        let before = record.mounts.len();
        record
            .mounts
            .retain(|(source, target, _)| source != &remove_tuple.0 || target != &remove_tuple.1);
        if record.mounts.len() < before {
            changes.push(format!(
                "removed volume: {}:{}",
                remove_tuple.0, remove_tuple.1
            ));
        }
    }
    for mount in &request.add_mounts {
        let tuple = mount.to_storage_tuple();
        if !record
            .mounts
            .iter()
            .any(|(source, target, _)| source == &tuple.0 && target == &tuple.1)
        {
            changes.push(format!(
                "added volume: {}:{}{}",
                tuple.0,
                tuple.1,
                if tuple.2 { ":ro" } else { "" }
            ));
            record.mounts.push(tuple);
        }
    }

    for remove in &request.remove_ports {
        let before = record.ports.len();
        record
            .ports
            .retain(|&(host, guest)| host != remove.host || guest != remove.guest);
        if record.ports.len() < before {
            changes.push(format!("removed port: {}:{}", remove.host, remove.guest));
        }
    }
    for port in &request.add_ports {
        let tuple = port.to_tuple();
        if !record.ports.contains(&tuple) {
            changes.push(format!("added port: {}:{}", tuple.0, tuple.1));
            record.ports.push(tuple);
        }
    }

    if let Some(cpus) = request.cpus {
        changes.push(format!("cpus: {} → {cpus}", record.cpus));
        record.cpus = cpus;
    }
    if let Some(memory_mib) = request.memory_mib {
        changes.push(format!("memory: {} MiB → {memory_mib} MiB", record.mem));
        record.mem = memory_mib;
    }

    if request.enable_network {
        changes.push("network: enabled".to_string());
        record.network = true;
    }
    if request.disable_network {
        changes.push("network: disabled".to_string());
        record.network = false;
        if record.allowed_cidrs.is_some() {
            changes.push("cleared allowed_cidrs".to_string());
            record.allowed_cidrs = None;
        }
        if record.dns_filter_hosts.is_some() {
            changes.push("cleared dns_filter_hosts".to_string());
            record.dns_filter_hosts = None;
        }
    }
    if let Some(network_backend) = request.network_backend {
        changes.push(format!("network_backend: {network_backend:?}"));
        record.network_backend = Some(network_backend);
    }
    if let Some(allowed_cidrs) = request.allowed_cidrs.clone() {
        changes.push("allowed_cidrs: replaced".to_string());
        record.allowed_cidrs = Some(allowed_cidrs);
        record.network = true;
    }
    if request.clear_allowed_cidrs {
        changes.push("allowed_cidrs: cleared".to_string());
        record.allowed_cidrs = None;
    }
    if let Some(hosts) = request.dns_filter_hosts.clone() {
        changes.push("dns_filter_hosts: replaced".to_string());
        record.dns_filter_hosts = Some(hosts);
        record.network = true;
    }
    if request.clear_dns_filter_hosts {
        changes.push("dns_filter_hosts: cleared".to_string());
        record.dns_filter_hosts = None;
    }

    for key in &request.remove_env {
        let before = record.env.len();
        record.env.retain(|(existing, _)| existing != key);
        if record.env.len() < before {
            changes.push(format!("removed env: {key}"));
        }
    }
    for (key, value) in &request.set_env {
        record.env.retain(|(existing, _)| existing != key);
        record.env.push((key.clone(), value.clone()));
        changes.push(format!("env: {key}=<redacted>"));
    }

    if request.clear_workdir {
        changes.push("workdir: cleared".to_string());
        record.workdir = None;
    }
    if let Some(workdir) = request.workdir.clone() {
        changes.push(format!("workdir: {workdir}"));
        record.workdir = Some(workdir);
    }

    if request.enable_gpu {
        changes.push("gpu: enabled".to_string());
        record.gpu = Some(true);
    }
    if request.disable_gpu {
        changes.push("gpu: disabled".to_string());
        record.gpu = Some(false);
    }
    if request.gpu_vram_mib.is_some() {
        changes.push(format!("gpu_vram_mib: {gpu_vram_mib:?}"));
        record.gpu_vram_mib = gpu_vram_mib;
    }

    if let Some(policy) = request.restart_policy.clone() {
        changes.push(format!("restart policy: {policy}"));
        record.restart.policy = policy;
    }
    if let Some(max_retries) = request.restart_max_retries {
        changes.push(format!("restart max_retries: {max_retries}"));
        record.restart.max_retries = max_retries;
    }
    if let Some(max_backoff_secs) = request.restart_max_backoff_secs {
        changes.push(format!("restart max_backoff_secs: {max_backoff_secs}"));
        record.restart.max_backoff_secs = max_backoff_secs;
    }

    if request.clear_health_cmd {
        changes.push("health command: cleared".to_string());
        record.health_cmd = None;
    }
    if let Some(health_cmd) = request.health_cmd.clone() {
        changes.push("health command: replaced".to_string());
        record.health_cmd = Some(health_cmd);
    }
    if let Some(value) = request.health_interval_secs {
        changes.push(format!("health interval: {value}s"));
        record.health_interval_secs = Some(value);
    }
    if let Some(value) = request.health_timeout_secs {
        changes.push(format!("health timeout: {value}s"));
        record.health_timeout_secs = Some(value);
    }
    if let Some(value) = request.health_retries {
        changes.push(format!("health retries: {value}"));
        record.health_retries = Some(value);
    }
    if let Some(value) = request.health_startup_grace_secs {
        changes.push(format!("health startup grace: {value}s"));
        record.health_startup_grace_secs = Some(value);
    }

    if request.enable_ssh_agent {
        changes.push("ssh_agent: enabled".to_string());
        record.ssh_agent = true;
    }
    if request.disable_ssh_agent {
        changes.push("ssh_agent: disabled".to_string());
        record.ssh_agent = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_image_info(entrypoint: Vec<&str>, cmd: Vec<&str>) -> ImageInfo {
        ImageInfo {
            reference: "alpine:latest".to_string(),
            digest: "sha256:test".to_string(),
            size: 0,
            created: None,
            architecture: "x86_64".to_string(),
            os: "linux".to_string(),
            layer_count: 0,
            layers: Vec::new(),
            entrypoint: entrypoint.into_iter().map(str::to_string).collect(),
            cmd: cmd.into_iter().map(str::to_string).collect(),
            env: Vec::new(),
            workdir: None,
            user: None,
        }
    }

    #[test]
    fn create_machine_builds_record() {
        let mut request = CreateMachine::new("dev");
        request.mounts.push(HostMount {
            source: PathBuf::from("/host"),
            target: PathBuf::from("/guest"),
            read_only: true,
        });
        request.ports.push(PortMapping {
            host: 2222,
            guest: 22,
        });
        let record = request.into_record().unwrap();
        assert_eq!(record.mounts, vec![("/host".into(), "/guest".into(), true)]);
        assert_eq!(record.ports, vec![(2222, 22)]);
    }

    #[test]
    fn run_session_command_prefers_explicit_command() {
        let mut request = MachineRun::new("run-test");
        request.command = vec!["echo".into(), "hi".into()];
        request.entrypoint = vec!["ignored".into()];
        let image_info = sample_image_info(vec!["also-ignored"], vec![]);
        assert_eq!(
            resolve_session_command(&request, Some(&image_info)),
            vec!["echo", "hi"]
        );
    }

    #[test]
    fn run_session_command_uses_config_before_image_defaults() {
        let mut request = MachineRun::new("run-test");
        request.entrypoint = vec!["/app/start".into()];
        request.cmd = vec!["--dev".into()];
        let image_info = sample_image_info(vec!["ignored"], vec!["ignored"]);
        assert_eq!(
            resolve_session_command(&request, Some(&image_info)),
            vec!["/app/start", "--dev"]
        );
    }

    #[test]
    fn run_session_command_falls_back_to_image_metadata() {
        let request = MachineRun::new("run-test");
        let image_info = sample_image_info(vec!["/entry"], vec!["arg"]);
        assert_eq!(
            resolve_session_command(&request, Some(&image_info)),
            vec!["/entry", "arg"]
        );
    }

    #[test]
    fn run_session_command_uses_shell_or_idle_when_empty() {
        let foreground = MachineRun::new("fg");
        assert_eq!(
            resolve_session_command(&foreground, None),
            vec![crate::DEFAULT_SHELL_CMD]
        );

        let mut detached = MachineRun::new("bg");
        detached.detached = true;
        assert_eq!(
            resolve_session_command(&detached, None),
            crate::DEFAULT_IDLE_CMD
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
        );
    }
}
