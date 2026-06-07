//! Machine management commands.
//!
//! All VM-related commands are under the `machine` subcommand:
//! - exec: Persistent execution (machine keeps running)
//! - create: Create named VM configuration
//! - start: Start a machine (named or default)
//! - stop: Stop a machine (named or default)
//! - delete: Delete a named VM configuration
//! - status: Show machine status
//! - ls: List all named VMs

use crate::cli::flush_output;
use crate::cli::format_bytes;
use crate::cli::parsers::{parse_cidr, parse_duration, parse_env_list, parse_image};
use crate::cli::vm_common;
use clap::{Args, Subcommand};
use smolvm::agent::{docker_config_mount, VmResources};
use smolvm::data::network::PortMapping;
use smolvm::data::resources::{DEFAULT_MICROVM_CPU_COUNT, DEFAULT_MICROVM_MEMORY_MIB};
use smolvm::data::storage::HostMount;
use smolvm::machine::{
    CreateMachine, DataDirMachine, DeleteMachine, DownloadMachineFile, ExecMachine, ForkMachine,
    GetMachine, ListMachineImages, LocalMachineService, MachineOperation, MachineService,
    NetworkTestMachine, PruneMachineImages, StartMachine, StopMachine, StorageStatusRequest,
    UpdateMachine, UploadMachineFile,
};
use smolvm::network::{validate_requested_network_backend, NetworkBackend};
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

/// Resolve `--allow-cidr`, `--allow-host`, and `--outbound-localhost-only` into a CIDR list,
/// net flag, and the original hostname list (for DNS filtering).
///
/// Resolution failure for `--allow-host` is a hard error — a typo or DNS outage
/// should not silently weaken the security policy.
/// Returns true when `s` structurally looks like an OCI image reference
/// rather than an executable name or path.
///
/// Catches the common mistake of writing `smolvm machine run ubuntu:22.04 --
/// bash` instead of `smolvm machine run --image ubuntu:22.04 -- bash`.
/// Only unambiguous structural signals are checked:
///   - `image:tag` form — colons are not valid in executable names
///   - `registry/image` or `namespace/image` form (non-absolute slash path)
///
/// Bare names like `alpine` or `nginx` are intentionally not flagged here
/// because they are indistinguishable from valid bare commands.
fn is_likely_image_ref(s: &str) -> bool {
    if s.contains(':') {
        return true;
    }
    s.contains('/') && !s.starts_with('/') && !s.starts_with("./") && !s.starts_with("../")
}

fn resolve_egress_flags(
    mut allow_cidr: Vec<String>,
    allow_host: Vec<String>,
    outbound_localhost_only: bool,
    net: bool,
) -> smolvm::Result<(Vec<String>, bool, Option<Vec<String>>)> {
    // Resolve hostnames to CIDRs — fail hard on resolution errors
    for host in &allow_host {
        let cidrs = crate::cli::parsers::resolve_host_to_cidrs(host)
            .map_err(|e| smolvm::Error::config("--allow-host", e))?;
        tracing::info!(host, ?cidrs, "resolved hostname for egress policy");
        allow_cidr.extend(cidrs);
    }

    if outbound_localhost_only {
        allow_cidr.push("127.0.0.0/8".to_string());
        allow_cidr.push("::1/128".to_string());
    }
    let net = net || !allow_cidr.is_empty();

    // Preserve original hostnames for DNS filtering (None if no --allow-host was used)
    let dns_filter_hosts = if allow_host.is_empty() {
        None
    } else {
        Some(allow_host)
    };

    Ok((allow_cidr, net, dns_filter_hosts))
}

/// Parse `--secret-env KEY=HOST_VAR` and `--secret-file KEY=PATH` flag values
/// into validated [`SecretRef`]s keyed by the guest-side env var name.
///
/// CLI-supplied refs are `TrustedLocal` (the host user invoked the command), so
/// both source kinds are allowed; `validate_ref` still enforces structure and
/// absolute `from_file` paths. A key that appears more than once — across or
/// within the two flags — is a hard error, since silently keeping the last
/// occurrence would mask a typo.
fn parse_cli_secret_refs(
    secret_env: &[String],
    secret_file: &[String],
) -> smolvm::Result<std::collections::BTreeMap<String, smolvm::secrets::SecretRef>> {
    use smolvm::secrets::{env_ref, file_ref, validate_ref, ResolutionScope, SecretRef};
    use std::collections::BTreeMap;

    let mut out: BTreeMap<String, SecretRef> = BTreeMap::new();

    let mut add =
        |flag: &str, spec: &str, make: &dyn Fn(&str) -> SecretRef| -> smolvm::Result<()> {
            let (key, value) = spec.split_once('=').ok_or_else(|| {
                smolvm::Error::config(flag, format!("expected KEY=VALUE, got '{}'", spec))
            })?;
            if key.is_empty() {
                return Err(smolvm::Error::config(
                    flag,
                    format!("empty secret name in '{}'", spec),
                ));
            }
            let r = make(value);
            validate_ref(&r, ResolutionScope::TrustedLocal)
                .map_err(|e| smolvm::Error::config(flag, format!("secret '{}': {}", key, e)))?;
            if out.insert(key.to_string(), r).is_some() {
                return Err(smolvm::Error::config(
                    flag,
                    format!("secret '{}' specified more than once", key),
                ));
            }
            Ok(())
        };

    for spec in secret_env {
        add("--secret-env", spec, &|v| env_ref(v))?;
    }
    for spec in secret_file {
        add("--secret-file", spec, &|v| file_ref(v))?;
    }
    Ok(out)
}

/// Manage machines
#[derive(Subcommand, Debug)]
pub enum MachineCmd {
    /// Run a container image in an ephemeral machine
    Run(RunCmd),

    /// Run a command directly in the VM (not in a container)
    Exec(ExecCmd),

    /// Create a new named machine configuration
    Create(CreateCmd),

    /// Start a machine
    Start(StartCmd),

    /// Fork a running forkable machine into a new clone (CoW memory + disks)
    Fork(ForkCmd),

    /// Stop a running machine
    Stop(StopCmd),

    /// Delete a machine configuration
    #[command(visible_alias = "rm")]
    Delete(DeleteCmd),

    /// Show machine status
    Status(StatusCmd),

    /// List all machines
    #[command(visible_alias = "list")]
    Ls(LsCmd),

    /// Resize a machine's disk resources (use `update` instead)
    #[command(hide = true)]
    Resize(ResizeCmd),

    /// Modify settings on a stopped machine (mounts, ports, resources, disks)
    Update(UpdateCmd),

    /// List cached images and storage usage
    Images(ImagesCmd),

    /// Remove unused images and layers to free disk space
    Prune(PruneCmd),

    /// Open an interactive shell in a machine (starts it if stopped)
    #[command(visible_alias = "sh")]
    Shell(ShellCmd),

    /// Copy files between host and machine
    Cp(CpCmd),

    /// Monitor a machine with health checks and restart policy
    Monitor(MonitorCmd),

    /// Test network connectivity from inside the VM
    #[command(hide = true)]
    NetworkTest(NetworkTestCmd),

    /// Print the on-disk data directory path for a named machine.
    ///
    /// Useful for scripting and debugging — returns the path where the VM's
    /// storage disk, overlay disk, and agent socket live. The path is
    /// hash-derived, not name-derived.
    #[command(name = "data-dir")]
    DataDir(DataDirCmd),
}

fn machine_operation_cli_binding(operation: MachineOperation) -> &'static str {
    match operation {
        MachineOperation::Create => "machine create",
        MachineOperation::Status => "machine status",
        MachineOperation::List => "machine ls",
        MachineOperation::Start => "machine start",
        MachineOperation::Stop => "machine stop",
        MachineOperation::Delete => "machine delete",
        MachineOperation::Fork => "machine fork",
        MachineOperation::Update => "machine update",
        MachineOperation::Exec => "machine exec",
        MachineOperation::ExecStream => "machine exec --stream",
        MachineOperation::ExecInteractive => "machine exec -it / machine shell",
        MachineOperation::Run => "machine run --image",
        MachineOperation::RunSession => "machine run",
        MachineOperation::Monitor => "machine monitor",
        MachineOperation::WriteFile => "machine cp (host to guest)",
        MachineOperation::UploadFile => "machine cp (host to guest file)",
        MachineOperation::ReadFile => "machine cp (guest to stdout/file)",
        MachineOperation::DownloadFile => "machine cp (guest to host file)",
        MachineOperation::StorageStatus => "machine images --storage",
        MachineOperation::ListImages => "machine images",
        MachineOperation::PullImage => "machine images --pull",
        MachineOperation::PruneImages => "machine prune",
        MachineOperation::NetworkTest => "machine network-test",
        MachineOperation::DataDir => "machine data-dir",
    }
}

impl MachineCmd {
    fn primary_operation(&self) -> MachineOperation {
        match self {
            MachineCmd::Run(_) => MachineOperation::RunSession,
            MachineCmd::Exec(cmd) if cmd.stream => MachineOperation::ExecStream,
            MachineCmd::Exec(cmd) if cmd.interactive || cmd.tty => {
                MachineOperation::ExecInteractive
            }
            MachineCmd::Exec(_) => MachineOperation::Exec,
            MachineCmd::Create(_) => MachineOperation::Create,
            MachineCmd::Start(_) => MachineOperation::Start,
            MachineCmd::Fork(_) => MachineOperation::Fork,
            MachineCmd::Stop(_) => MachineOperation::Stop,
            MachineCmd::Delete(_) => MachineOperation::Delete,
            MachineCmd::Status(_) => MachineOperation::Status,
            MachineCmd::Ls(_) => MachineOperation::List,
            MachineCmd::Resize(_) | MachineCmd::Update(_) => MachineOperation::Update,
            MachineCmd::Images(_) => MachineOperation::ListImages,
            MachineCmd::Prune(_) => MachineOperation::PruneImages,
            MachineCmd::Shell(_) => MachineOperation::ExecInteractive,
            MachineCmd::Cp(_) => MachineOperation::DownloadFile,
            MachineCmd::Monitor(_) => MachineOperation::Monitor,
            MachineCmd::NetworkTest(_) => MachineOperation::NetworkTest,
            MachineCmd::DataDir(_) => MachineOperation::DataDir,
        }
    }

    pub fn run(self) -> smolvm::Result<()> {
        let _ = machine_operation_cli_binding(self.primary_operation());
        // Skip orphan cleanup for ephemeral `machine run` — it creates and
        // immediately destroys its VM, so stale records don't affect it.
        // Other commands (ls, exec, create, etc.) clean up first.
        if !matches!(self, MachineCmd::Run(_)) {
            super::vm_common::cleanup_orphaned_ephemeral_vms();
        }

        match self {
            MachineCmd::Run(cmd) => cmd.run(),
            MachineCmd::Exec(cmd) => cmd.run(),
            MachineCmd::Create(cmd) => cmd.run(),
            MachineCmd::Start(cmd) => cmd.run(),
            MachineCmd::Fork(cmd) => cmd.run(),
            MachineCmd::Stop(cmd) => cmd.run(),
            MachineCmd::Delete(cmd) => cmd.run(),
            MachineCmd::Status(cmd) => cmd.run(),
            MachineCmd::Ls(cmd) => cmd.run(),
            MachineCmd::Resize(cmd) => cmd.run(),
            MachineCmd::Update(cmd) => cmd.run(),
            MachineCmd::Images(cmd) => cmd.run(),
            MachineCmd::Prune(cmd) => cmd.run(),
            MachineCmd::Shell(cmd) => cmd.run(),
            MachineCmd::Cp(cmd) => cmd.run(),
            MachineCmd::Monitor(cmd) => cmd.run(),
            MachineCmd::NetworkTest(cmd) => cmd.run(),
            MachineCmd::DataDir(cmd) => cmd.run(),
        }
    }
}

// ============================================================================
// Run Command (Ephemeral)
// ============================================================================

/// Run a container image in an ephemeral machine.
///
/// By default, runs in ephemeral mode (machine cleaned up after exit).
/// Use -d/--detach to keep the machine running for later interaction.
///
/// Examples:
///   smolvm machine run --image alpine -- echo "hello"
///   smolvm machine run -it -I alpine
///   smolvm machine run -d --net -I ubuntu
///   smolvm machine run --net -v ./src:/app --image node -- npm start
#[derive(Args, Debug)]
pub struct RunCmd {
    /// Container image (e.g., alpine, ubuntu:22.04, ghcr.io/org/image).
    /// Optional when a Smolfile provides the image, or for bare VM mode.
    #[arg(short = 'I', long, value_name = "IMAGE", value_parser = parse_image)]
    pub image: Option<String>,

    /// Run a packed `.smolmachine` artifact ephemerally (the VM is discarded on
    /// exit) — the one-shot equivalent of `machine create --from … + start`.
    /// CPU/memory fall back to the artifact's baked manifest unless overridden.
    #[arg(
        long,
        value_name = "PATH",
        conflicts_with_all = ["image", "smolfile", "detach", "name", "gpu", "gpu_vram_mib", "oci_platform", "allow_cidr", "allow_host", "outbound_localhost_only", "secret_env", "secret_file"],
        help_heading = "Machine source"
    )]
    pub from: Option<PathBuf>,

    /// Name a persistent machine when used with --detach.
    /// Matches the --name flag on start/stop/exec/status/resize. In foreground
    /// mode (no -d), --name is ignored with a warning.
    #[arg(short = 'n', long, value_name = "NAME", help_heading = "Execution")]
    pub name: Option<String>,

    /// Command and arguments to run (default: image entrypoint or /bin/sh)
    #[arg(trailing_var_arg = true, value_name = "COMMAND")]
    pub command: Vec<String>,

    /// Start the command in the background and detach, leaving the VM
    /// running. Use `machine exec` to run further commands against the VM
    /// and `machine stop` to tear it down.
    #[arg(short = 'd', long, help_heading = "Execution")]
    pub detach: bool,

    /// Keep stdin open for interactive input
    #[arg(short = 'i', long, help_heading = "Execution")]
    pub interactive: bool,

    /// Allocate a pseudo-TTY (use with -i for interactive shells)
    #[arg(short = 't', long, help_heading = "Execution")]
    pub tty: bool,

    /// Kill command after duration (e.g., "30s", "5m", "1h")
    #[arg(long, value_parser = parse_duration, value_name = "DURATION", help_heading = "Execution")]
    pub timeout: Option<Duration>,

    /// Set working directory inside container
    #[arg(short = 'w', long, value_name = "DIR", help_heading = "Container")]
    pub workdir: Option<String>,

    /// Set environment variable (can be used multiple times)
    #[arg(
        short = 'e',
        long = "env",
        value_name = "KEY=VALUE",
        help_heading = "Container"
    )]
    pub env: Vec<String>,

    /// Target OCI platform for multi-arch images
    #[arg(
        long = "oci-platform",
        value_name = "OS/ARCH",
        help_heading = "Container"
    )]
    pub oci_platform: Option<String>,

    /// Mount host directory into container (can be used multiple times)
    #[arg(
        short = 'v',
        long = "volume",
        value_name = "HOST:CONTAINER[:ro]",
        help_heading = "Container"
    )]
    pub volume: Vec<String>,

    /// Expose port from container to host (can be used multiple times)
    #[arg(short = 'p', long = "port", value_parser = PortMapping::parse, value_name = "HOST:GUEST", help_heading = "Network")]
    pub port: Vec<PortMapping>,

    /// Enable outbound network access
    #[arg(long, help_heading = "Network")]
    pub net: bool,

    /// Select the networking backend.
    #[arg(
        long = "net-backend",
        value_enum,
        hide = true,
        help_heading = "Network"
    )]
    pub net_backend: Option<NetworkBackend>,

    /// Allow egress to specific CIDR range (can be used multiple times, implies --net)
    #[arg(long = "allow-cidr", value_parser = parse_cidr, value_name = "CIDR", help_heading = "Network")]
    pub allow_cidr: Vec<String>,

    /// Allow egress to specific hostname, resolved at VM start (can be used multiple times, implies --net)
    #[arg(long = "allow-host", value_name = "HOSTNAME", help_heading = "Network")]
    pub allow_host: Vec<String>,

    /// Restrict outbound to localhost only (implies --net)
    #[arg(long, help_heading = "Network")]
    pub outbound_localhost_only: bool,

    /// Enable GPU acceleration (Vulkan via virtio-gpu)
    #[arg(long, help_heading = "Resources")]
    pub gpu: bool,

    /// GPU shared-memory region size in MiB. Ignored without --gpu.
    /// Default 4096 (4 GiB). Must be > 0.
    #[arg(
        long = "gpu-vram",
        value_name = "MiB",
        help_heading = "Resources",
        value_parser = crate::cli::parsers::parse_gpu_vram_mib,
    )]
    pub gpu_vram_mib: Option<u32>,

    /// Number of virtual CPUs
    #[arg(long, default_value_t = DEFAULT_MICROVM_CPU_COUNT, value_name = "N", help_heading = "Resources")]
    pub cpus: u8,

    /// Memory allocation in MiB
    #[arg(long, default_value_t = DEFAULT_MICROVM_MEMORY_MIB, value_name = "MiB", help_heading = "Resources")]
    pub mem: u32,

    /// Storage disk size in GiB
    #[arg(long, value_name = "GiB", help_heading = "Resources")]
    pub storage: Option<u64>,

    /// Overlay disk size in GiB
    #[arg(long, value_name = "GiB", help_heading = "Resources")]
    pub overlay: Option<u64>,

    /// Load VM configuration from a Smolfile (TOML)
    #[arg(
        long = "smolfile",
        visible_short_alias = 's',
        value_name = "PATH",
        help_heading = "Resources"
    )]
    pub smolfile: Option<PathBuf>,

    /// Forward host SSH agent into the VM (enables git/ssh without exposing keys)
    #[arg(long, help_heading = "Security")]
    pub ssh_agent: bool,

    /// Mount ~/.docker/ config into VM for registry authentication
    #[arg(long, help_heading = "Registry")]
    pub docker_config: bool,

    /// Inject a secret from a host env var (GUEST_VAR=HOST_VAR), resolved at
    /// launch. The value is never persisted to the machine record or a pack.
    #[arg(
        long = "secret-env",
        value_name = "GUEST_VAR=HOST_VAR",
        help_heading = "Security"
    )]
    pub secret_env: Vec<String>,

    /// Inject a secret from a host file (GUEST_VAR=/abs/path), resolved at
    /// launch. The value is never persisted to the machine record or a pack.
    #[arg(
        long = "secret-file",
        value_name = "GUEST_VAR=PATH",
        help_heading = "Security"
    )]
    pub secret_file: Vec<String>,

    #[command(flatten, next_help_heading = "Network")]
    pub proxy_opts: crate::cli::proxy_opts::ProxyOpts,
}

impl RunCmd {
    pub fn run(self) -> smolvm::Result<()> {
        use smolvm::machine::{MachineRun, MachineRunIo, MachineRunPullProgress, MachineRunResult};
        use smolvm::Error;

        // `--from`: run a packed .smolmachine artifact ephemerally, reusing the
        // proven pack-run path. Resource flags fall back to the artifact's baked
        // manifest values (matching `machine create --from`); the remaining run
        // flags pass through. Flags the sidecar runner can't honor are rejected
        // at parse time via `conflicts_with_all` on `from`.
        if let Some(from) = self.from {
            return crate::cli::pack_run::PackRunCmd {
                sidecar: Some(from),
                command: self.command,
                interactive: self.interactive,
                tty: self.tty,
                timeout: self.timeout,
                workdir: self.workdir,
                env: self.env,
                volume: self.volume,
                port: self.port,
                net: self.net,
                net_backend: self.net_backend,
                cpus: (self.cpus != DEFAULT_MICROVM_CPU_COUNT).then_some(self.cpus),
                mem: (self.mem != DEFAULT_MICROVM_MEMORY_MIB).then_some(self.mem),
                storage: self.storage,
                overlay: self.overlay,
                force_extract: false,
                info: false,
                debug: false,
            }
            .run();
        }

        let requested_name = self.name.clone();
        let explicit_name = requested_name.is_some();
        let vm_name = if self.detach {
            requested_name.unwrap_or_else(|| "default".to_string())
        } else {
            smolvm::util::generate_machine_name()
        };

        if explicit_name && vm_name != "default" && self.detach {
            let service = LocalMachineService::new()?;
            if service.status(GetMachine::new(vm_name.clone()))?.is_some() {
                return Err(Error::config(
                    "machine run -d --name",
                    format!(
                        "a machine named '{}' already exists. Use 'machine start --name {}' to start it, or 'machine delete {} -f' to remove it.",
                        vm_name, vm_name, vm_name
                    ),
                ));
            }
        }

        let cli_command = self.command.clone();
        let (cli_allow_cidrs, net, cli_dns_filter_hosts) = resolve_egress_flags(
            self.allow_cidr,
            self.allow_host,
            self.outbound_localhost_only,
            self.net,
        )?;

        let params = crate::cli::smolfile::build_create_params(
            vm_name.clone(),
            self.image.clone(),
            None,
            cli_command.clone(),
            self.cpus,
            self.mem,
            self.volume,
            self.port,
            net,
            self.net_backend,
            vec![],
            self.env,
            self.workdir,
            self.smolfile,
            self.storage,
            self.overlay,
            cli_allow_cidrs,
        )?;

        let mut params = params;
        params.dns_filter_hosts = match (params.dns_filter_hosts.take(), cli_dns_filter_hosts) {
            (Some(mut from_smolfile), Some(mut from_cli)) => {
                from_smolfile.append(&mut from_cli);
                Some(from_smolfile)
            }
            (Some(from_smolfile), None) => Some(from_smolfile),
            (None, some) => some,
        };
        for (key, r) in parse_cli_secret_refs(&self.secret_env, &self.secret_file)? {
            params.secret_refs.insert(key, r);
        }

        let mut mounts = HostMount::parse(&params.volume)?;
        if self.docker_config {
            if let Some(docker_mount) = docker_config_mount() {
                mounts.push(docker_mount);
            } else {
                tracing::warn!("Docker config directory not found");
            }
        }
        PortMapping::check_duplicates(&params.port)
            .map_err(|e| smolvm::Error::config("validate ports", e))?;

        if self.detach && (self.interactive || self.tty) {
            eprintln!("warning: -i/-t flags are ignored in detached mode (-d)");
        }

        let has_smolfile_command = !params.entrypoint.is_empty() || !params.cmd.is_empty();
        let (interactive, tty) = if !self.interactive
            && !self.tty
            && !self.detach
            && cli_command.is_empty()
            && !has_smolfile_command
        {
            return Err(smolvm::Error::config(
                "machine run",
                "no command specified.\n\
                     Use: smolvm machine run -- <command>\n\
                     Or:  smolvm machine run -it",
            ));
        } else {
            (self.interactive, self.tty)
        };

        {
            let resolved_image = self.image.as_deref().or(params.image.as_deref());
            if resolved_image.is_none()
                && !cli_command.is_empty()
                && is_likely_image_ref(&cli_command[0])
            {
                let cmd0 = &cli_command[0];
                let rest: Vec<&str> = cli_command[1..]
                    .iter()
                    .filter(|s| s.as_str() != "--")
                    .map(|s| s.as_str())
                    .collect();
                let suggestion = if rest.is_empty() {
                    format!("smolvm machine run --image {cmd0}")
                } else {
                    format!("smolvm machine run --image {cmd0} -- {}", rest.join(" "))
                };
                return Err(Error::config(
                    "machine run",
                    format!(
                        "'{cmd0}' looks like a container image reference, not a command.\n\
                         To run a container, use --image:\n  {suggestion}"
                    ),
                ));
            }
        }

        let resources = VmResources {
            cpus: params.cpus,
            memory_mib: params.mem,
            network: params.net,
            network_backend: params.network_backend,
            gpu: self.gpu || params.gpu,
            gpu_vram_mib: self.gpu_vram_mib.or(params.gpu_vram_mib),
            storage_gib: params.storage_gb,
            overlay_gib: params.overlay_gb,
            allowed_cidrs: params.allowed_cidrs.clone(),
        };
        resources.validate()?;
        validate_requested_network_backend(
            &resources,
            params.dns_filter_hosts.as_deref(),
            params.port.len(),
        )?;

        if self.detach {
            eprintln!("Starting persistent machine...");
        } else {
            eprintln!("Starting ephemeral machine ({})...", vm_name);
        }

        let mut request = MachineRun::new(vm_name.clone());
        request.detached = self.detach;
        // Preserve legacy behavior: an explicit non-default detached name must
        // be new, while implicit `default` may update the existing default record.
        request.allow_existing_record = self.detach && !(explicit_name && vm_name != "default");
        request.image = self.image.clone().or(params.image.clone());
        request.command = cli_command;
        request.entrypoint = params.entrypoint.clone();
        request.cmd = params.cmd.clone();
        request.env = parse_env_list(&params.env);
        request.secret_refs = params.secret_refs.clone();
        request.workdir = params.workdir.clone();
        request.mounts = mounts;
        request.ports = params.port.clone();
        request.resources = resources;
        request.init = params.init.clone();
        request.ssh_agent = self.ssh_agent || params.ssh_agent;
        request.dns_filter_hosts = params.dns_filter_hosts.clone();
        request.oci_platform = self.oci_platform.clone();
        request.proxy = self.proxy_opts.proxy().map(str::to_string);
        request.no_proxy = self.proxy_opts.no_proxy().map(str::to_string);
        request.timeout = self.timeout;
        request.interactive = interactive;
        request.tty = tty;
        request.kill_on_sigint = true;

        struct CliRunIo {
            last_percent: u8,
            syncing: bool,
        }

        impl MachineRunIo for CliRunIo {
            fn pull_progress(&mut self, event: &MachineRunPullProgress) {
                if event.layer == "syncing" {
                    if !self.syncing {
                        eprint!(
                            "\rPulling image {}... [====================] 100% — syncing...",
                            event.image
                        );
                        let _ = std::io::stderr().flush();
                        self.syncing = true;
                    }
                    return;
                }
                let percent = event.current as u8;
                if percent != self.last_percent && percent <= 100 {
                    eprint!("\rPulling image {}... [", event.image);
                    let filled = (percent as usize) / 5;
                    for i in 0..20 {
                        if i < filled {
                            eprint!("=");
                        } else if i == filled {
                            eprint!(">");
                        } else {
                            eprint!(" ");
                        }
                    }
                    eprint!("] {}%", percent);
                    let _ = std::io::stderr().flush();
                    self.last_percent = percent;
                }
            }

            fn stdout(&mut self, bytes: &[u8]) {
                let _ = std::io::stdout().write_all(bytes);
            }

            fn stderr(&mut self, bytes: &[u8]) {
                let _ = std::io::stderr().write_all(bytes);
            }
        }

        if let Some(image) = request.image.as_ref() {
            eprint!("Pulling image {}...", image);
            let _ = std::io::stderr().flush();
        }
        let mut io = CliRunIo {
            last_percent: 0,
            syncing: false,
        };
        let result = LocalMachineService::new()?.run_session(request, &mut io)?;
        if let MachineRunResult::Detached { name, pid } = result {
            if name == "default" {
                println!("Machine running in background");
                if let Some(pid) = pid {
                    tracing::debug!(pid, "machine running in background");
                }
                println!("\nTo interact:");
                println!("  smolvm machine exec -- <command>");
                println!("\nTo stop:");
                println!("  smolvm machine stop");
            } else {
                println!("Machine '{}' running in background", name);
                println!("\nTo interact:");
                println!("  smolvm machine exec --name {} -- <command>", name);
                println!("\nTo stop:");
                println!("  smolvm machine stop --name {}", name);
            }
            Ok(())
        } else if let MachineRunResult::Foreground { exit_code, .. } = result {
            flush_output();
            std::process::exit(exit_code);
        } else {
            unreachable!("run_session returned an unknown result variant")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parse_cli_secret_refs_builds_env_and_file_refs() {
        let refs = parse_cli_secret_refs(
            &["GUEST_TOKEN=HOST_TOKEN".to_string()],
            &["GUEST_KEY=/abs/key".to_string()],
        )
        .unwrap();
        assert_eq!(refs["GUEST_TOKEN"].from_env.as_deref(), Some("HOST_TOKEN"));
        assert_eq!(
            refs["GUEST_KEY"]
                .from_file
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned()),
            Some("/abs/key".to_string())
        );
    }

    #[test]
    fn parse_cli_secret_refs_rejects_bad_specs() {
        // Missing '='.
        assert!(parse_cli_secret_refs(&["NO_EQUALS".to_string()], &[]).is_err());
        // Empty key.
        assert!(parse_cli_secret_refs(&["=HOST".to_string()], &[]).is_err());
        // Relative from_file path (validate_ref under TrustedLocal).
        assert!(parse_cli_secret_refs(&[], &["K=relative/path".to_string()]).is_err());
        // Duplicate key across the two flags.
        assert!(
            parse_cli_secret_refs(&["DUP=HOST".to_string()], &["DUP=/abs/path".to_string()])
                .is_err()
        );
    }

    #[derive(Parser, Debug)]
    #[command(name = "machine")]
    struct TestMachineCli {
        #[command(subcommand)]
        command: MachineCmd,
    }

    #[test]
    fn run_detach_accepts_name_flag() {
        let cli = TestMachineCli::parse_from([
            "machine", "run", "-d", "--name", "foo", "--image", "alpine",
        ]);

        let MachineCmd::Run(cmd) = cli.command else {
            panic!("expected machine run command");
        };
        assert_eq!(cmd.name, Some("foo".to_string()));
        assert!(cmd.detach);
    }

    // Documents the clap parsing behaviour: positionals before "--" land in
    // `command`, not `image`.  is_likely_image_ref() catches the unambiguous
    // cases before a VM is booted.
    #[test]
    fn run_image_ref_as_positional_lands_in_command_vec() {
        let cli = TestMachineCli::parse_from(["machine", "run", "ubuntu:22.04", "--", "bash"]);
        let MachineCmd::Run(cmd) = cli.command else {
            panic!("expected machine run command");
        };
        assert_eq!(cmd.image, None);
        // With trailing_var_arg, clap includes the "--" separator in the vec.
        assert_eq!(cmd.command, ["ubuntu:22.04", "--", "bash"]);
        // is_likely_image_ref catches this before the VM starts
        assert!(is_likely_image_ref(&cmd.command[0]));
    }

    #[test]
    fn create_accepts_trailing_workload_command() {
        let cli = TestMachineCli::parse_from([
            "machine", "create", "golden", "--image", "alpine", "--", "echo", "hi",
        ]);
        let MachineCmd::Create(cmd) = cli.command else {
            panic!("expected machine create command");
        };
        assert_eq!(cmd.name, Some("golden".to_string()));
        assert_eq!(cmd.image, Some("alpine".to_string()));
        // The trailing command is captured (clap may include the "--" separator).
        let words: Vec<&str> = cmd
            .command
            .iter()
            .map(String::as_str)
            .filter(|s| *s != "--")
            .collect();
        assert_eq!(words, ["echo", "hi"]);
    }

    #[test]
    fn create_without_command_leaves_command_empty() {
        // Regression: adding the trailing COMMAND arg must not break the common
        // no-command form `machine create <name> --net`.
        let cli = TestMachineCli::parse_from(["machine", "create", "golden", "--net"]);
        let MachineCmd::Create(cmd) = cli.command else {
            panic!("expected machine create command");
        };
        assert_eq!(cmd.name, Some("golden".to_string()));
        assert!(cmd.command.is_empty());
        assert!(cmd.net);
    }

    #[test]
    fn is_likely_image_ref_classifies_correctly() {
        // Unambiguous image references
        assert!(is_likely_image_ref("ubuntu:22.04")); // image:tag
        assert!(is_likely_image_ref("ghcr.io/org/image")); // registry/path
        assert!(is_likely_image_ref("library/alpine")); // namespace/image

        // Bare names are not flagged — indistinguishable from commands at parse time
        assert!(!is_likely_image_ref("alpine"));
        assert!(!is_likely_image_ref("bash"));

        // Absolute and relative paths are always commands
        assert!(!is_likely_image_ref("/bin/sh"));
        assert!(!is_likely_image_ref("./script.sh"));
    }
}

// ============================================================================
// Exec Command (Persistent) - Direct VM Execution
// ============================================================================

/// Execute a command directly in the VM's Alpine rootfs.
///
/// This runs commands at the VM level, not inside a container. Useful for
/// debugging, inspecting the VM environment, or running VM-level operations.
///
/// Examples:
///   smolvm machine exec -- uname -a
///   smolvm machine exec --name myvm -- df -h
///   smolvm machine exec -it -- /bin/sh
#[derive(Args, Debug)]
pub struct ExecCmd {
    /// Command and arguments to execute
    #[arg(trailing_var_arg = true, required = true, value_name = "COMMAND")]
    pub command: Vec<String>,

    /// Target machine (default: "default")
    #[arg(long, value_name = "NAME")]
    pub name: Option<String>,

    /// Set working directory in the VM
    #[arg(short = 'w', long, value_name = "DIR")]
    pub workdir: Option<String>,

    /// Set environment variable (can be used multiple times)
    #[arg(short = 'e', long = "env", value_name = "KEY=VALUE")]
    pub env: Vec<String>,

    /// Inject a secret from a host env var (GUEST_VAR=HOST_VAR) for this exec,
    /// resolved on the host. The value never persists to the record.
    #[arg(long = "secret-env", value_name = "GUEST_VAR=HOST_VAR")]
    pub secret_env: Vec<String>,

    /// Inject a secret from a host file (GUEST_VAR=/abs/path) for this exec,
    /// resolved on the host. The value never persists to the record.
    #[arg(long = "secret-file", value_name = "GUEST_VAR=PATH")]
    pub secret_file: Vec<String>,

    /// Kill command after duration (e.g., "30s", "5m")
    #[arg(long, value_parser = parse_duration, value_name = "DURATION")]
    pub timeout: Option<Duration>,

    /// Keep stdin open for interactive input
    #[arg(short = 'i', long)]
    pub interactive: bool,

    /// Allocate a pseudo-TTY (use with -i for shells)
    #[arg(short = 't', long)]
    pub tty: bool,

    /// Stream output in real-time (prints as it arrives)
    #[arg(long)]
    pub stream: bool,
}

impl ExecCmd {
    pub fn run(self) -> smolvm::Result<()> {
        let name =
            vm_common::resolve_vm_name(self.name.clone())?.unwrap_or_else(|| "default".to_string());
        let mut request = ExecMachine::new(name, self.command.clone());
        request.env = parse_env_list(&self.env);
        request.secret_refs = parse_cli_secret_refs(&self.secret_env, &self.secret_file)?;
        request.secret_scope = smolvm::secrets::ResolutionScope::TrustedLocal;
        request.workdir = self.workdir.clone();
        request.timeout = self.timeout;
        request.tty = self.tty;
        request.start_if_needed = false;

        let service = LocalMachineService::new()?;
        if self.interactive || self.tty {
            let exit_code = service.exec_interactive(request)?;
            std::process::exit(exit_code);
        }

        if self.stream {
            let mut printer = ExecEventPrinter::default();
            let exit_code = service.exec_stream(request, &mut |event| printer.handle(event))?;
            std::process::exit(exit_code);
        }

        let result = service.exec(request)?;
        if !result.stdout.is_empty() {
            let _ = std::io::stdout().write_all(&result.stdout);
        }
        if !result.stderr.is_empty() {
            let _ = std::io::stderr().write_all(&result.stderr);
        }
        flush_output();
        std::process::exit(result.exit_code);
    }
}

#[derive(Default)]
struct ExecEventPrinter {
    exit_code: i32,
}

impl ExecEventPrinter {
    fn handle(&mut self, event: smolvm::agent::ExecEvent) {
        match event {
            smolvm::agent::ExecEvent::Stdout(data) => {
                let _ = std::io::stdout().write_all(&data);
                let _ = std::io::stdout().flush();
            }
            smolvm::agent::ExecEvent::Stderr(data) => {
                let _ = std::io::stderr().write_all(&data);
                let _ = std::io::stderr().flush();
            }
            smolvm::agent::ExecEvent::Exit(code) => {
                self.exit_code = code;
            }
            smolvm::agent::ExecEvent::Error(msg) => {
                eprintln!("error: {}", msg);
                self.exit_code = 1;
            }
        }
    }
}

// ============================================================================
// Shell Command
// ============================================================================

/// Open an interactive shell in a machine.
///
/// Shortcut for `machine exec -it -- /bin/sh`. Starts the machine if stopped.
///
/// Examples:
///   smolvm machine shell
///   smolvm machine shell --name myvm
///   smolvm machine sh --name myvm
#[derive(Args, Debug)]
pub struct ShellCmd {
    /// Target machine (default: "default")
    #[arg(long, short = 'n', value_name = "NAME")]
    pub name: Option<String>,
}

impl ShellCmd {
    pub fn run(self) -> smolvm::Result<()> {
        // Delegate to exec with -it -- /bin/sh
        ExecCmd {
            command: vec!["/bin/sh".to_string()],
            name: self.name,
            workdir: None,
            env: vec![],
            secret_env: vec![],
            secret_file: vec![],
            timeout: None,
            interactive: true,
            tty: true,
            stream: false,
        }
        .run()
    }
}

// ============================================================================
// Create Command
// ============================================================================

/// Create a named machine configuration.
///
/// Creates a persistent VM configuration that can be started later.
/// Use `smolvm machine start --name <name>` to start, then
/// `smolvm machine exec --name <name> -- <command>` to run commands inside.
///
/// Examples:
///   smolvm machine create myvm
///   smolvm machine create webserver --cpus 2 --mem 1024 -p 80:80
#[derive(Args, Debug)]
pub struct CreateCmd {
    /// Name for the machine (auto-generated if omitted)
    #[arg(value_name = "NAME")]
    pub name: Option<String>,

    /// Container image (e.g., alpine, python:3.12-alpine)
    #[arg(short = 'I', long, value_name = "IMAGE", value_parser = parse_image)]
    pub image: Option<String>,

    /// Number of virtual CPUs
    #[arg(long, default_value_t = DEFAULT_MICROVM_CPU_COUNT, value_name = "N")]
    pub cpus: u8,

    /// Memory allocation in MiB
    #[arg(long, default_value_t = DEFAULT_MICROVM_MEMORY_MIB, value_name = "MiB")]
    pub mem: u32,

    /// Storage disk size in GiB (for OCI layers and container data)
    #[arg(long, value_name = "GiB")]
    pub storage: Option<u64>,

    /// Overlay disk size in GiB (for persistent rootfs changes)
    #[arg(long, value_name = "GiB")]
    pub overlay: Option<u64>,

    /// Mount host directory (can be used multiple times)
    #[arg(short = 'v', long = "volume", value_name = "HOST:GUEST[:ro]")]
    pub volume: Vec<String>,

    /// Expose port from VM to host (can be used multiple times)
    #[arg(short = 'p', long = "port", value_parser = PortMapping::parse, value_name = "HOST:GUEST")]
    pub port: Vec<PortMapping>,

    /// Enable outbound network access
    #[arg(long)]
    pub net: bool,

    /// Select the networking backend.
    #[arg(long = "net-backend", value_enum, hide = true)]
    pub net_backend: Option<NetworkBackend>,

    /// Allow egress to specific CIDR range (can be used multiple times, implies --net)
    #[arg(long = "allow-cidr", value_parser = parse_cidr, value_name = "CIDR")]
    pub allow_cidr: Vec<String>,

    /// Allow egress to specific hostname, resolved at VM start (can be used multiple times, implies --net)
    #[arg(long = "allow-host", value_name = "HOSTNAME")]
    pub allow_host: Vec<String>,

    /// Restrict outbound to localhost only (implies --net)
    #[arg(long)]
    pub outbound_localhost_only: bool,

    /// Enable GPU acceleration (Vulkan via virtio-gpu)
    #[arg(long)]
    pub gpu: bool,

    /// GPU shared-memory region size in MiB. Ignored without --gpu.
    /// Default 4096 (4 GiB). Must be > 0.
    #[arg(
        long = "gpu-vram",
        value_name = "MiB",
        value_parser = crate::cli::parsers::parse_gpu_vram_mib,
    )]
    pub gpu_vram_mib: Option<u32>,

    /// Run command on every VM start (can be used multiple times)
    #[arg(long = "init", value_name = "COMMAND")]
    pub init: Vec<String>,

    /// Set environment variable (can be used multiple times)
    #[arg(short = 'e', long = "env", value_name = "KEY=VALUE")]
    pub env: Vec<String>,

    /// Set working directory inside the machine
    #[arg(short = 'w', long = "workdir", value_name = "DIR")]
    pub workdir: Option<String>,

    /// Forward host SSH agent into the VM (enables git/ssh without exposing keys)
    #[arg(long)]
    pub ssh_agent: bool,

    /// Inject a secret from a host env var (GUEST_VAR=HOST_VAR), resolved at
    /// each launch. Only the reference is persisted, never the value.
    #[arg(long = "secret-env", value_name = "GUEST_VAR=HOST_VAR")]
    pub secret_env: Vec<String>,

    /// Inject a secret from a host file (GUEST_VAR=/abs/path), resolved at
    /// each launch. Only the reference is persisted, never the value.
    #[arg(long = "secret-file", value_name = "GUEST_VAR=PATH")]
    pub secret_file: Vec<String>,

    /// Load configuration from a Smolfile (TOML)
    #[arg(long = "smolfile", visible_short_alias = 's', value_name = "PATH")]
    pub smolfile: Option<PathBuf>,

    /// Create machine from a packed .smolmachine artifact.
    /// Uses pre-extracted layers instead of pulling from a registry.
    #[arg(long, value_name = "PATH", conflicts_with_all = ["image", "smolfile"])]
    pub from: Option<PathBuf>,

    /// Command to run as the machine's persistent workload (image machines).
    /// Launched as a detached container on every `start`, so it stays running
    /// (e.g. a pre-warmed browser to be forked). Without this, an image machine
    /// boots to a bare agent and the image's CMD is not run.
    #[arg(trailing_var_arg = true, value_name = "COMMAND")]
    pub command: Vec<String>,
}

impl CreateCmd {
    pub fn run(self) -> smolvm::Result<()> {
        // Branch for --from: create machine from .smolmachine artifact.
        if let Some(ref sidecar_path) = self.from {
            return self.run_from_smolmachine(sidecar_path);
        }

        let (cli_allow_cidrs, net, cli_dns_filter_hosts) = resolve_egress_flags(
            self.allow_cidr,
            self.allow_host,
            self.outbound_localhost_only,
            self.net,
        )?;

        let name = self
            .name
            .unwrap_or_else(smolvm::util::generate_machine_name);

        let params = crate::cli::smolfile::build_create_params(
            name,
            self.image,
            None,         // entrypoint: from Smolfile only
            self.command, // persistent-workload command (detached container on start)
            self.cpus,
            self.mem,
            self.volume,
            self.port,
            net,
            self.net_backend,
            self.init,
            self.env,
            self.workdir,
            self.smolfile,
            self.storage,
            self.overlay,
            cli_allow_cidrs,
        )?;
        let mut params = params;
        params.dns_filter_hosts = match (params.dns_filter_hosts.take(), cli_dns_filter_hosts) {
            (Some(mut from_smolfile), Some(mut from_cli)) => {
                from_smolfile.append(&mut from_cli);
                Some(from_smolfile)
            }
            (Some(from_smolfile), None) => Some(from_smolfile),
            (None, some) => some,
        };
        // CLI `--secret-env`/`--secret-file` refs merge over any Smolfile
        // `[secrets]` of the same name (CLI wins). Only refs are persisted.
        for (key, r) in parse_cli_secret_refs(&self.secret_env, &self.secret_file)? {
            params.secret_refs.insert(key, r);
        }
        let resources = VmResources {
            cpus: params.cpus,
            memory_mib: params.mem,
            network: params.net,
            network_backend: params.network_backend,
            gpu: params.gpu,
            gpu_vram_mib: params.gpu_vram_mib,
            storage_gib: params.storage_gb,
            overlay_gib: params.overlay_gb,
            allowed_cidrs: params.allowed_cidrs.clone(),
        };
        // Reject zero-valued resources before the machine is persisted.
        // Without this, `machine create` succeeds and the failure only
        // surfaces later at `machine start` (see QA BUG-44).
        resources.validate()?;
        validate_requested_network_backend(
            &resources,
            params.dns_filter_hosts.as_deref(),
            params.port.len(),
        )?;
        if self.ssh_agent {
            params.ssh_agent = true;
        }
        if self.gpu {
            params.gpu = true;
        }
        // CLI --gpu-vram takes precedence over Smolfile gpu_vram.
        if let Some(vram) = self.gpu_vram_mib {
            params.gpu_vram_mib = Some(vram);
        }
        PortMapping::check_duplicates(&params.port)
            .map_err(|e| smolvm::Error::config("validate ports", e))?;
        vm_common::create_vm(params)
    }

    /// Create a machine from a .smolmachine artifact.
    fn run_from_smolmachine(&self, sidecar_path: &std::path::Path) -> smolvm::Result<()> {
        use smolvm::data::resources::{DEFAULT_MICROVM_CPU_COUNT, DEFAULT_MICROVM_MEMORY_MIB};

        if !sidecar_path.exists() {
            return Err(smolvm::Error::config(
                "create from .smolmachine",
                format!("file not found: {}", sidecar_path.display()),
            ));
        }

        // Read manifest from the sidecar to get image metadata.
        let manifest = smolvm_pack::packer::read_manifest_from_sidecar(sidecar_path)
            .map_err(|e| smolvm::Error::agent("read .smolmachine", e.to_string()))?;

        // Resolve the canonical path for storage in VmRecord.
        let canonical_path = sidecar_path
            .canonicalize()
            .unwrap_or_else(|_| sidecar_path.to_path_buf())
            .to_string_lossy()
            .into_owned();

        let name = self
            .name
            .clone()
            .unwrap_or_else(smolvm::util::generate_machine_name);
        // CLI flags override manifest defaults.
        let cpus = if self.cpus != DEFAULT_MICROVM_CPU_COUNT {
            self.cpus
        } else {
            manifest.cpus
        };
        let mem = if self.mem != DEFAULT_MICROVM_MEMORY_MIB {
            self.mem
        } else {
            manifest.mem
        };

        // A .smolmachine is an untrusted, portable artifact: validate its secret
        // refs under the Untrusted scope, which rejects every source kind. A
        // packed `from_env`/`from_file` ref would otherwise read THIS host's
        // env/files at exec time — reject at create rather than carry an exfil
        // primitive. Configure secrets locally via the CLI instead.
        for (key, r) in &manifest.secret_refs {
            smolvm::secrets::validate_ref(r, smolvm::secrets::ResolutionScope::Untrusted).map_err(
                |e| {
                    smolvm::Error::config(
                        "create from .smolmachine",
                        format!("secret '{}': {} (packs may not carry secret refs)", key, e),
                    )
                },
            )?;
        }

        let params = vm_common::CreateVmParams {
            secret_refs: manifest.secret_refs,
            name,
            image: Some(manifest.image),
            entrypoint: manifest.entrypoint,
            cmd: manifest.cmd,
            cpus,
            mem,
            volume: self.volume.clone(),
            port: self.port.clone(),
            net: self.net || manifest.network,
            network_backend: self.net_backend,
            init: self.init.clone(),
            env: {
                let mut env = manifest.env;
                env.extend(self.env.iter().cloned());
                env
            },
            workdir: manifest.workdir,
            storage_gb: self.storage,
            overlay_gb: self.overlay,
            allowed_cidrs: None,
            restart_policy: None,
            restart_max_retries: None,
            restart_max_backoff_secs: None,
            health_cmd: None,
            health_interval_secs: None,
            health_timeout_secs: None,
            health_retries: None,
            health_startup_grace_secs: None,
            ssh_agent: self.ssh_agent,
            dns_filter_hosts: None,
            gpu: manifest.gpu,
            gpu_vram_mib: None,
            source_smolmachine: Some(canonical_path),
        };

        vm_common::create_vm(params)
    }
}

// ============================================================================
// Start Command
// ============================================================================

/// Start a machine.
///
/// Starts the VM process. If no name is given, starts the default VM.
#[derive(Args, Debug)]
pub struct StartCmd {
    /// Machine to start (default: "default")
    #[arg(short = 'n', long, value_name = "NAME")]
    pub name: Option<String>,

    /// Start as a fork base: back guest RAM with a memfd (CoW-cloneable) and
    /// expose a control socket so the machine can be forked with `machine fork`.
    #[arg(long)]
    pub forkable: bool,

    #[command(flatten, next_help_heading = "Network")]
    pub proxy_opts: crate::cli::proxy_opts::ProxyOpts,
}

impl StartCmd {
    pub fn run(self) -> smolvm::Result<()> {
        let explicit_name = self.name.is_some();
        let name = self.name.unwrap_or_else(|| "default".to_string());
        let mut request = StartMachine::new(name.clone());
        request.forkable = self.forkable;
        request.proxy = self.proxy_opts.proxy().map(str::to_string);
        request.no_proxy = self.proxy_opts.no_proxy().map(str::to_string);
        match LocalMachineService::new()?.start(request) {
            Ok(_) => Ok(()),
            Err(smolvm::Error::VmNotFound { .. }) if !explicit_name => {
                let service = LocalMachineService::new()?;
                let _ = service.create(CreateMachine::new("default"))?;
                let mut request = StartMachine::new("default");
                request.proxy = self.proxy_opts.proxy().map(str::to_string);
                request.no_proxy = self.proxy_opts.no_proxy().map(str::to_string);
                let _ = service.start(request)?;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}

// ============================================================================
// Fork Command
// ============================================================================

/// Fork a running forkable machine into a new clone.
///
/// Freezes the source (the "golden") via its control socket, copy-on-write
/// clones its disks, and boots the new machine from the golden's in-memory
/// snapshot instead of cold-booting — so the clone comes up already warm
/// (same processes, same filesystem state), in well under a second.
///
/// The golden must have been started with `--forkable`.
#[derive(Args, Debug)]
pub struct ForkCmd {
    /// The running, forkable source machine to clone from.
    #[arg(value_name = "GOLDEN")]
    pub golden: String,

    /// Name for the new clone machine.
    #[arg(value_name = "CLONE")]
    pub clone: String,

    /// Make the clone itself forkable (memfd RAM + control socket), so it can
    /// in turn be forked.
    #[arg(long)]
    pub forkable: bool,

    /// Pin the clone's inbound port forwards (repeatable). Without this, the
    /// golden's forwards are remapped to freshly-allocated host ports.
    #[arg(short = 'p', long = "port", value_parser = PortMapping::parse, value_name = "HOST:GUEST", help_heading = "Network")]
    pub port: Vec<PortMapping>,
}

impl ForkCmd {
    pub fn run(self) -> smolvm::Result<()> {
        if self.forkable {
            return Err(smolvm::Error::agent(
                "fork",
                "nested fork is not supported: a clone cannot be re-forked, so `--forkable` has no effect (drop it)",
            ));
        }
        let mut request = ForkMachine::new(self.golden.clone(), self.clone.clone());
        request.ports = self.port.clone();
        let status = LocalMachineService::new()?.fork(request)?;
        eprintln!(
            "Forked '{}' -> '{}'. Golden stays frozen as the fork base (do not start it again while clones exist).",
            self.golden, status.name
        );
        Ok(())
    }
}

// ============================================================================
// Stop Command
// ============================================================================

/// Stop a running machine.
///
/// Gracefully stops the VM process. Running containers will be terminated.
#[derive(Args, Debug)]
pub struct StopCmd {
    /// Machine to stop (default: "default")
    #[arg(short = 'n', long, value_name = "NAME")]
    pub name: Option<String>,
}

impl StopCmd {
    pub fn run(self) -> smolvm::Result<()> {
        let name = vm_common::resolve_vm_name(self.name)?.unwrap_or_else(|| "default".to_string());
        LocalMachineService::new()?.stop(StopMachine::new(name))?;
        Ok(())
    }
}

// ============================================================================
// Delete Command
// ============================================================================

/// Delete a machine configuration.
///
/// Removes the VM configuration. Does not delete container data.
#[derive(Args, Debug)]
pub struct DeleteCmd {
    /// Machine to delete
    #[arg(value_name = "NAME")]
    pub name: String,

    /// Skip confirmation prompt
    #[arg(short, long)]
    pub force: bool,
}

impl DeleteCmd {
    pub fn run(&self) -> smolvm::Result<()> {
        if !self.force {
            print!("Delete machine '{}' and all its data? [y/N] ", self.name);
            let _ = std::io::stdout().flush();
            let mut answer = String::new();
            std::io::stdin().read_line(&mut answer).ok();
            if !matches!(answer.trim(), "y" | "Y" | "yes" | "YES") {
                println!("Aborted.");
                return Ok(());
            }
        }
        let mut request = DeleteMachine::new(self.name.clone());
        request.break_dependent_clones = self.force;
        LocalMachineService::new()?.delete(request)?;
        println!("Deleted machine: {}", self.name);
        Ok(())
    }
}

// ============================================================================
// Status Command
// ============================================================================

/// Show machine status.
///
/// Displays whether the VM is running and its process ID.
#[derive(Args, Debug)]
pub struct StatusCmd {
    /// Machine to check (default: "default")
    #[arg(short = 'n', long, value_name = "NAME")]
    pub name: Option<String>,

    /// Output in JSON format
    #[arg(long)]
    pub json: bool,
}

impl StatusCmd {
    pub fn run(self) -> smolvm::Result<()> {
        if self.json {
            return vm_common::status_vm_json(&self.name);
        }
        vm_common::status_vm(&self.name, |_| {})
    }
}

// ============================================================================
// Ls Command
// ============================================================================

/// List all machines.
///
/// Shows all configured VMs with their state, resources, and configuration.
#[derive(Args, Debug)]
pub struct LsCmd {
    /// Show detailed configuration (mounts, ports, PID)
    #[arg(short, long)]
    pub verbose: bool,

    /// Output in JSON format
    #[arg(long)]
    pub json: bool,
}

impl LsCmd {
    pub fn run(&self) -> smolvm::Result<()> {
        vm_common::list_vms(self.verbose, self.json)
    }
}

// ============================================================================
// Resize Command
// ============================================================================

/// Resize a machine's disk resources.
///
/// Expands the storage and/or overlay disk for a stopped machine.
/// The VM must be stopped before resizing. Disk expansion happens
/// immediately; filesystem resize occurs automatically on next boot.
///
/// Examples:
///   smolvm machine resize --name my-vm --storage 50
///   smolvm machine resize --name my-vm --overlay 20
///   smolvm machine resize --name my-vm --storage 50 --overlay 20
///   smolvm machine resize --storage 50  # default VM
#[derive(Args, Debug)]
#[command(group(
    clap::ArgGroup::new("resize-target")
        .required(true)
        .args(["storage", "overlay"])
        .multiple(true)
))]
pub struct ResizeCmd {
    /// Machine to resize (default: "default")
    #[arg(short = 'n', long, value_name = "NAME")]
    pub name: Option<String>,

    /// Storage disk size in GiB (expand only)
    #[arg(long, value_name = "GiB")]
    pub storage: Option<u64>,

    /// Overlay disk size in GiB (expand only)
    #[arg(long, value_name = "GiB")]
    pub overlay: Option<u64>,
}

impl ResizeCmd {
    pub fn run(self) -> smolvm::Result<()> {
        let name = vm_common::resolve_vm_name(self.name)?;
        let name_str = name.as_deref().unwrap_or("default");

        vm_common::resize_vm(name_str, self.storage, self.overlay).map_err(|e| {
            if matches!(&e, smolvm::Error::InvalidState { .. }) {
                smolvm::Error::agent(
                    "resize",
                    format!(
                        "VM '{}' is running. Stop it first with: smolvm machine stop --name {}",
                        name_str, name_str
                    ),
                )
            } else {
                e
            }
        })
    }
}

// ============================================================================
// Update Command
// ============================================================================

/// Modify settings on a stopped machine.
///
/// Changes are applied to the DB record and take effect on the next
/// `machine start`. The machine must be stopped.
///
/// Examples:
///   smolvm machine update myvm -v ./src:/app -p 8080:8080
///   smolvm machine update myvm --cpus 4 --mem 4096
///   smolvm machine update myvm --remove-volume ./src:/app
///   smolvm machine update myvm --net -e DEBUG=1
#[derive(Args, Debug)]
pub struct UpdateCmd {
    /// Machine to update
    #[arg(value_name = "NAME")]
    pub name: String,

    /// Add volume mount (HOST:GUEST[:ro])
    #[arg(short = 'v', long = "volume", value_name = "HOST:GUEST[:ro]")]
    pub volume: Vec<String>,

    /// Remove volume mount (HOST:GUEST)
    #[arg(long, value_name = "HOST:GUEST")]
    pub remove_volume: Vec<String>,

    /// Add port mapping (HOST:GUEST)
    #[arg(short = 'p', long = "port", value_parser = PortMapping::parse, value_name = "HOST:GUEST")]
    pub port: Vec<PortMapping>,

    /// Remove port mapping (HOST:GUEST)
    #[arg(long, value_parser = PortMapping::parse, value_name = "HOST:GUEST")]
    pub remove_port: Vec<PortMapping>,

    /// Set vCPU count
    #[arg(long, value_name = "N")]
    pub cpus: Option<u8>,

    /// Set memory in MiB
    #[arg(long, value_name = "MiB")]
    pub mem: Option<u32>,

    /// Enable outbound network access
    #[arg(long)]
    pub net: bool,

    /// Disable outbound network access
    #[arg(long, conflicts_with = "net")]
    pub no_net: bool,

    /// Add/replace environment variable (KEY=VALUE)
    #[arg(short = 'e', long = "env", value_name = "KEY=VALUE")]
    pub env: Vec<String>,

    /// Remove environment variable by key
    #[arg(long, value_name = "KEY")]
    pub remove_env: Vec<String>,

    /// Set working directory
    #[arg(short = 'w', long, value_name = "DIR")]
    pub workdir: Option<String>,

    /// Enable GPU acceleration
    #[arg(long)]
    pub gpu: bool,

    /// Disable GPU acceleration
    #[arg(long, conflicts_with = "gpu")]
    pub no_gpu: bool,

    /// Storage disk size in GiB (expand only)
    #[arg(long, value_name = "GiB")]
    pub storage: Option<u64>,

    /// Overlay disk size in GiB (expand only)
    #[arg(long, value_name = "GiB")]
    pub overlay: Option<u64>,
}

impl UpdateCmd {
    pub fn run(self) -> smolvm::Result<()> {
        let mut request = UpdateMachine::new(self.name.clone());
        request.add_mounts = HostMount::parse(&self.volume)?;
        request.remove_mounts = HostMount::parse(&self.remove_volume)?;
        request.add_ports = self.port.clone();
        request.remove_ports = self.remove_port.clone();
        request.cpus = self.cpus;
        request.memory_mib = self.mem;
        request.enable_network = self.net;
        request.disable_network = self.no_net;
        request.set_env = parse_env_list(&self.env);
        request.remove_env = self.remove_env.clone();
        request.workdir = self.workdir.clone();
        request.enable_gpu = self.gpu;
        request.disable_gpu = self.no_gpu;
        request.storage_gb = self.storage;
        request.overlay_gb = self.overlay;

        let result = LocalMachineService::new()?.update(request)?;
        if result.changes.is_empty() {
            println!("No changes specified.");
        } else {
            println!("Updated machine '{}':", self.name);
            for change in &result.changes {
                println!("  {}", change);
            }
            println!("\nStart with: smolvm machine start --name {}", self.name);
        }
        Ok(())
    }
}

// ============================================================================
// Data Dir Command
// ============================================================================

/// Print the on-disk data directory for a named machine.
///
/// Equivalent to calling `smolvm::agent::vm_data_dir(name)` — exposed as a
/// CLI command so shell scripts and external tooling have a single source
/// of truth for the path computation (which is hash-derived, not
/// name-derived).
#[derive(Args, Debug)]
pub struct DataDirCmd {
    /// Machine name.
    pub name: String,
}

impl DataDirCmd {
    pub fn run(self) -> smolvm::Result<()> {
        let dir = LocalMachineService::new()?.data_dir(DataDirMachine::new(self.name))?;
        println!("{}", dir.display());
        Ok(())
    }
}

// ============================================================================
// Network Test Command
// ============================================================================

/// Test network connectivity directly from machine (debug TSI).
#[derive(Args, Debug)]
pub struct NetworkTestCmd {
    /// Named machine to test (omit for default)
    #[arg(long)]
    pub name: Option<String>,

    /// URL to test
    #[arg(default_value = "http://1.1.1.1")]
    pub url: String,
}

impl NetworkTestCmd {
    pub fn run(self) -> smolvm::Result<()> {
        let name =
            vm_common::resolve_vm_name(self.name.clone())?.unwrap_or_else(|| "default".to_string());
        println!("Testing network from machine: {}", self.url);
        let mut request = NetworkTestMachine::new(name, self.url);
        request.start_if_needed = true;
        let result = LocalMachineService::new()?.network_test(request)?;
        println!(
            "Result: {}",
            serde_json::to_string_pretty(&result).unwrap_or_default()
        );
        Ok(())
    }
}

// ============================================================================
// Images Command
// ============================================================================

/// List cached images and storage usage.
///
/// Shows all OCI images cached in the machine's storage, along with their
/// sizes and layer counts. Also displays total storage usage.
///
/// Examples:
///   smolvm machine images --name myvm
///   smolvm machine images --name myvm --json
#[derive(Args, Debug)]
pub struct ImagesCmd {
    /// Machine to query
    #[arg(long, required = true, value_name = "NAME")]
    pub name: String,

    /// Output in JSON format
    #[arg(long)]
    pub json: bool,
}

impl ImagesCmd {
    pub fn run(self) -> smolvm::Result<()> {
        let service = LocalMachineService::new()?;
        let initial_state = service
            .status(GetMachine::new(self.name.clone()))?
            .ok_or_else(|| smolvm::Error::vm_not_found(&self.name))?
            .state;
        let started_for_query = initial_state != smolvm::config::RecordState::Running;
        if started_for_query {
            eprintln!("Starting machine '{}' to query storage...", self.name);
        }

        let mut storage_request = StorageStatusRequest::new(self.name.clone());
        storage_request.start_if_needed = true;
        let status = service.storage_status(storage_request)?;
        let mut images_request = ListMachineImages::new(self.name.clone());
        images_request.start_if_needed = true;
        let images = service.list_images(images_request)?;

        if self.json {
            let output = serde_json::json!({
                "storage": {
                    "total_bytes": status.total_bytes,
                    "used_bytes": status.used_bytes,
                    "layer_count": status.layer_count,
                    "image_count": status.image_count,
                },
                "images": images,
            });
            let json = serde_json::to_string_pretty(&output)
                .map_err(|e| smolvm::Error::config("serialize json", e.to_string()))?;
            println!("{}", json);
        } else {
            println!("Storage Usage:");
            println!("  Total:  {}", format_bytes(status.total_bytes));
            println!("  Used:   {}", format_bytes(status.used_bytes));
            println!("  Layers: {}", status.layer_count);
            println!();

            if images.is_empty() {
                println!("No cached images.");
            } else {
                println!("Cached Images:");
                println!("{:<40} {:>10} {:>8}", "IMAGE", "SIZE", "LAYERS");
                println!("{}", "-".repeat(60));

                for image in &images {
                    let name = if image.reference.len() > 38 {
                        format!("{}...", &image.reference[..35])
                    } else {
                        image.reference.clone()
                    };
                    println!(
                        "{:<40} {:>10} {:>8}",
                        name,
                        format_bytes(image.size),
                        image.layer_count
                    );
                }

                println!();
                println!("Total: {} images", images.len());
            }
        }

        if started_for_query {
            let _ = service.stop(StopMachine::new(self.name));
        }

        Ok(())
    }
}

// ============================================================================
// Prune Command
// ============================================================================

/// Remove unused images and layers to free disk space.
///
/// This removes layers that are not referenced by any cached image manifest.
/// Use --dry-run to see what would be removed without actually deleting.
///
/// Examples:
///   smolvm machine prune --name myvm --dry-run
///   smolvm machine prune --name myvm
///   smolvm machine prune --name myvm --all
#[derive(Args, Debug)]
pub struct PruneCmd {
    /// Machine to prune
    #[arg(long, required = true, value_name = "NAME")]
    pub name: String,

    /// Show what would be removed without actually removing
    #[arg(long)]
    pub dry_run: bool,

    /// Remove all cached images (not just unreferenced layers)
    #[arg(long)]
    pub all: bool,
}

impl PruneCmd {
    pub fn run(self) -> smolvm::Result<()> {
        let service = LocalMachineService::new()?;
        let initial_state = service
            .status(GetMachine::new(self.name.clone()))?
            .ok_or_else(|| smolvm::Error::vm_not_found(&self.name))?
            .state;
        if initial_state != smolvm::config::RecordState::Running {
            eprintln!("Starting machine...");
        }

        let mut request = PruneMachineImages::new(self.name.clone());
        request.dry_run = self.dry_run;
        request.all = self.all;
        request.stop_after_start = true;
        let result = service.prune_images(request)?;

        if self.all {
            if result.removed_images == 0 && result.freed_bytes == 0 {
                println!("No cached images to remove.");
            } else if self.dry_run {
                println!(
                    "Would remove {} images ({})",
                    result.removed_images,
                    format_bytes(result.freed_bytes)
                );
            } else {
                println!(
                    "Removed {} images, freed {}",
                    result.removed_images,
                    format_bytes(result.freed_bytes)
                );
            }
        } else if self.dry_run {
            if result.freed_bytes > 0 {
                println!(
                    "Would free {} of unreferenced layers",
                    format_bytes(result.freed_bytes)
                );
            } else {
                println!("No unreferenced layers to remove.");
            }
        } else if result.freed_bytes > 0 {
            println!("Freed {}", format_bytes(result.freed_bytes));
        } else {
            println!("No unreferenced layers to remove.");
        }

        Ok(())
    }
}

// ============================================================================
// Cp (File Copy) Command
// ============================================================================

/// Copy files between host and a running machine.
///
/// Uses `machine:path` syntax to specify the remote side.
///
/// Examples:
///   smolvm machine cp ./script.py myvm:/workspace/script.py    # upload
///   smolvm machine cp myvm:/workspace/output.json ./output.json # download
#[derive(Args, Debug)]
pub struct CpCmd {
    /// Source path (local file or machine:path)
    #[arg(value_name = "SRC")]
    pub src: String,

    /// Destination path (local file or machine:path)
    #[arg(value_name = "DST")]
    pub dst: String,
}

impl CpCmd {
    pub fn run(self) -> smolvm::Result<()> {
        let (machine_name, guest_path, local_path, is_upload) =
            if let Some((name, path)) = self.src.split_once(':') {
                (name.to_string(), path.to_string(), self.dst.clone(), false)
            } else if let Some((name, path)) = self.dst.split_once(':') {
                (name.to_string(), path.to_string(), self.src.clone(), true)
            } else {
                return Err(smolvm::Error::config(
                    "cp",
                    "one of SRC or DST must use machine:path syntax (e.g., myvm:/workspace/file)",
                ));
            };

        let service = LocalMachineService::new()?;
        if is_upload {
            let result = service.upload_file(UploadMachineFile::new(
                machine_name,
                PathBuf::from(&local_path),
                guest_path,
            ))?;
            eprintln!("Uploaded {} bytes", result.bytes);
        } else {
            let result = service.download_file(DownloadMachineFile::new(
                machine_name,
                guest_path,
                PathBuf::from(&local_path),
            ))?;
            eprintln!("Downloaded {} bytes", result.bytes);
        }

        Ok(())
    }
}

// ============================================================================
// Monitor Command
// ============================================================================

/// Monitor a running machine with health checks and restart policy.
///
/// Runs in the foreground, watching the machine and restarting on crash
/// or health check failure. Uses the restart policy from the machine's
/// config (set via Smolfile [restart] or --restart flag on create).
///
/// Ctrl+C stops monitoring; the machine keeps running.
///
/// Examples:
///   smolvm machine monitor --name myvm
///   smolvm machine monitor --name myvm --health-cmd "curl -f http://localhost:8080/health"
///   smolvm machine monitor --name myvm --restart always --interval 10
#[derive(Args, Debug)]
pub struct MonitorCmd {
    /// Machine to monitor (default: "default")
    #[arg(short = 'n', long, value_name = "NAME")]
    pub name: Option<String>,

    /// Override restart policy (never, always, on-failure, unless-stopped)
    #[arg(long, value_name = "POLICY")]
    pub restart: Option<String>,

    /// Health check command (run inside the VM via sh -c)
    #[arg(long, value_name = "CMD")]
    pub health_cmd: Option<String>,

    /// Health check timeout in seconds
    #[arg(long, default_value = "5", value_name = "SECS")]
    pub health_timeout: u64,

    /// Check interval in seconds
    #[arg(long, default_value = "5", value_name = "SECS")]
    pub interval: u64,

    /// Health check failures before triggering restart
    #[arg(long, default_value = "3", value_name = "N")]
    pub health_retries: u32,
}

impl MonitorCmd {
    pub fn run(self) -> smolvm::Result<()> {
        use smolvm::config::RestartPolicy;
        use smolvm::machine::{MonitorEvent, MonitorMachine};
        use smolvm::Error;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let name = self.name.unwrap_or_else(|| "default".to_string());
        let restart_policy = self
            .restart
            .as_deref()
            .map(str::parse::<RestartPolicy>)
            .transpose()
            .map_err(|e| Error::config("--restart", e))?;

        let mut request = MonitorMachine::new(name.clone());
        request.restart_policy = restart_policy;
        request.health_cmd = self
            .health_cmd
            .clone()
            .map(|command| vec!["sh".into(), "-c".into(), command]);
        request.health_timeout = Duration::from_secs(self.health_timeout);
        request.interval = Duration::from_secs(self.interval);
        request.health_retries = self.health_retries;

        // Ctrl+C handler via SIGINT.
        //
        // SAFETY: `stop` is an Arc<AtomicBool> that lives until the end of this
        // function. The cloned Arc below keeps a strong reference alive for the
        // duration of the monitor loop, so the raw pointer stored in STOP_FLAG
        // remains valid until after the loop exits and the function returns. The
        // handler only does an atomic store, which is async-signal-safe.
        let stop = Arc::new(AtomicBool::new(false));
        {
            let stop = stop.clone();
            unsafe {
                let _ = libc::signal(libc::SIGINT, {
                    static mut STOP_FLAG: *const AtomicBool = std::ptr::null();
                    STOP_FLAG = Arc::as_ptr(&stop);
                    extern "C" fn handler(_: libc::c_int) {
                        unsafe {
                            if !STOP_FLAG.is_null() {
                                (*STOP_FLAG).store(true, Ordering::SeqCst);
                            }
                        }
                    }
                    handler as *const () as libc::sighandler_t
                });
            }
        }

        let mut on_event = |event: MonitorEvent| match event {
            MonitorEvent::Starting { name } => {
                println!("Machine '{}' is not running, starting...", name);
            }
            MonitorEvent::Monitoring {
                name,
                policy,
                interval_secs,
                health,
            } => {
                println!(
                    "Monitoring machine '{}' (policy: {}, interval: {}s)",
                    name, policy, interval_secs
                );
                if let Some(health) = health {
                    println!(
                        "  Health check: retries={}, timeout={}s",
                        health.retries, health.timeout_secs
                    );
                }
            }
            MonitorEvent::SuspendDetected { sleep_secs } => {
                println!(
                    "  detected suspend (~{}s) — skipping health check for recovery",
                    sleep_secs
                );
            }
            MonitorEvent::HealthRecovered => {
                println!("  health check passed (recovered)");
            }
            MonitorEvent::HealthFailed {
                exit_code,
                consecutive,
                retries,
                stderr,
            } => {
                println!(
                    "  health check failed (exit {}, {}/{}): {}",
                    exit_code, consecutive, retries, stderr
                );
            }
            MonitorEvent::HealthError {
                consecutive,
                retries,
                error,
            } => {
                println!(
                    "  health check error ({}/{}): {}",
                    consecutive, retries, error
                );
            }
            MonitorEvent::AgentUnreachable {
                consecutive,
                retries,
            } => {
                println!("  cannot connect to agent ({}/{})", consecutive, retries);
            }
            MonitorEvent::UnhealthyStopping => {
                println!("  unhealthy — stopping machine for restart");
            }
            MonitorEvent::MachineExited { exit_code } => {
                println!(
                    "  machine exited (exit code: {})",
                    exit_code
                        .map(|code| code.to_string())
                        .unwrap_or_else(|| "unknown".into())
                );
            }
            MonitorEvent::Restarting {
                attempt,
                backoff_secs,
            } => {
                println!(
                    "  restarting (attempt {}, backoff {}s)...",
                    attempt, backoff_secs
                );
            }
            MonitorEvent::Restarted => {
                println!("  machine restarted");
            }
            MonitorEvent::RestartFailed { error } => {
                println!("  restart failed: {}", error);
            }
            MonitorEvent::NotRestarting {
                policy,
                count,
                max_retries,
            } => {
                println!(
                    "  not restarting (policy: {}, count: {}/{})",
                    policy,
                    count,
                    if max_retries > 0 {
                        max_retries.to_string()
                    } else {
                        "unlimited".into()
                    }
                );
            }
            MonitorEvent::Stopped { name } => {
                println!(
                    "\nStopped monitoring. Machine '{}' may still be running.",
                    name
                );
            }
            _ => {}
        };

        LocalMachineService::new()?
            .monitor(request, &mut on_event, &|| stop.load(Ordering::SeqCst))?;
        Ok(())
    }
}
