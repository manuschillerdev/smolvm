//! Internal boot subprocess for the smolvm CLI.
//!
//! This command is NOT for direct user invocation. It's spawned to launch a VM
//! in a fresh single-threaded process, avoiding the macOS fork-in-multithreaded-process issue.

/// Run the boot subprocess.
pub fn run(config_path: std::path::PathBuf) -> smolvm::Result<()> {
    smolvm::boot::run(config_path)
}
