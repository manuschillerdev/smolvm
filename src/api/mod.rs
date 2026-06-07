//! HTTP API server for smolvm.
//!
//! This module provides an HTTP API for managing machines, containers, and images
//! without CLI overhead.
//!
//! # Example
//!
//! ```bash
//! # Start the server on the default Unix socket
//! smolvm serve start
//!
//! # Or start the server on TCP explicitly
//! smolvm serve start --listen 127.0.0.1:8080
//!
//! # Create a machine
//! curl -X POST http://localhost:8080/api/v1/machines \
//!   -H "Content-Type: application/json" \
//!   -d '{"name": "test"}'
//! ```

#[path = "errors.rs"]
pub mod error;
pub mod handlers;
pub mod state;
pub mod supervisor;
pub mod types;

use axum::{
    extract::Request,
    http::{HeaderValue, StatusCode},
    middleware::{self, Next},
    response::Response,
    routing::{delete, get, patch, post, put, MethodRouter},
    Router,
};
use std::sync::Arc;
use std::time::Duration;
use tower_http::{
    cors::{AllowOrigin, CorsLayer},
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use self::error::ApiError;
use crate::machine::MachineOperation;
use state::ApiState;

/// OpenAPI documentation for the smolvm API.
#[derive(OpenApi)]
#[openapi(
    info(
        title = "smolvm API",
        version = "0.5.2",
        description = "smolvm API for managing machines and images.",
        license(name = "Apache-2.0", url = "https://www.apache.org/licenses/LICENSE-2.0")
    ),
    tags(
        (name = "Health", description = "Health check endpoints"),
        (name = "Node", description = "Node capacity introspection"),
        (name = "Machines", description = "Machine lifecycle management"),
        (name = "Execution", description = "Command execution in machines"),
        (name = "Logs", description = "Log streaming"),
        (name = "Images", description = "OCI image management"),
        (name = "Files", description = "File upload and download")
    ),
    paths(
        // Health
        handlers::health::health,
        // Node
        handlers::node::capacity,
        // Execution
        handlers::exec::exec_command,
        handlers::exec::exec_stream,
        handlers::exec::run_command,
        handlers::exec::run_session,
        handlers::exec::stream_logs,
        // Files
        handlers::files::upload_file,
        handlers::files::download_file,
        // Images
        handlers::images::list_images,
        handlers::images::pull_image,
        handlers::images::prune_images,
        handlers::images::storage_status,
        // Machines
        handlers::machines::create_machine,
        handlers::machines::list_machines,
        handlers::machines::get_machine,
        handlers::machines::start_machine,
        handlers::machines::fork_machine,
        handlers::machines::stop_machine,
        handlers::machines::delete_machine,
        handlers::machines::update_machine,
        handlers::machines::resize_machine,
        handlers::machines::monitor_machine,
        handlers::machines::network_test,
        handlers::machines::data_dir,
    ),
    components(schemas(
        // Request types
        types::CreateMachineRequest,
        types::StartMachineRequest,
        types::ForkMachineRequest,
        types::UpdateMachineRequest,
        types::RestartSpec,
        types::MountSpec,
        types::PortSpec,
        types::ResourceSpec,
        types::ExecRequest,
        types::RunRequest,
        types::MachineRunRequest,
        types::EnvVar,
        types::PullImageRequest,
        types::PruneImagesRequest,
        types::NetworkTestRequest,
        types::MonitorQuery,
        types::DeleteQuery,
        types::LogsQuery,
        types::ResizeMachineRequest,
        // Response types
        types::HealthResponse,
        types::CapacityResponse,
        types::MachineInfo,
        types::MountInfo,
        types::ListMachinesResponse,
        types::ExecResponse,
        types::ImageInfo,
        types::ListImagesResponse,
        types::PullImageResponse,
        types::PruneImagesResponse,
        types::StorageStatusResponse,
        types::NetworkTestResponse,
        types::DataDirResponse,
        types::MachineRunResponse,
        types::StartResponse,
        types::StopResponse,
        types::DeleteResponse,
        types::ApiErrorResponse,
    ))
)]
pub struct ApiDoc;

/// Default timeout for API requests (5 minutes).
/// Most operations (start, stop, exec) complete within this time.
/// Long-running operations like image pulls may need longer, but this
/// provides a reasonable upper bound for most requests.
const API_REQUEST_TIMEOUT_SECS: u64 = 300;

/// Validate that an API command payload is not empty.
pub fn validate_command(cmd: &[String]) -> Result<(), ApiError> {
    if cmd.is_empty() {
        return Err(ApiError::BadRequest("command cannot be empty".into()));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HttpRouteGroup {
    Timed,
    LongLived,
}

enum HttpOperationBinding {
    Route {
        group: HttpRouteGroup,
        path: &'static str,
    },
    CoveredBy {
        operation: MachineOperation,
    },
}

fn machine_operation_http_binding(operation: MachineOperation) -> HttpOperationBinding {
    match operation {
        MachineOperation::Create => HttpOperationBinding::Route {
            group: HttpRouteGroup::Timed,
            path: "/",
        },
        MachineOperation::Status => HttpOperationBinding::Route {
            group: HttpRouteGroup::Timed,
            path: "/{id}",
        },
        MachineOperation::List => HttpOperationBinding::Route {
            group: HttpRouteGroup::Timed,
            path: "/",
        },
        MachineOperation::Start => HttpOperationBinding::Route {
            group: HttpRouteGroup::Timed,
            path: "/{id}/start",
        },
        MachineOperation::Stop => HttpOperationBinding::Route {
            group: HttpRouteGroup::Timed,
            path: "/{id}/stop",
        },
        MachineOperation::Delete => HttpOperationBinding::Route {
            group: HttpRouteGroup::Timed,
            path: "/{id}",
        },
        MachineOperation::Fork => HttpOperationBinding::Route {
            group: HttpRouteGroup::Timed,
            path: "/{id}/fork",
        },
        MachineOperation::Update => HttpOperationBinding::Route {
            group: HttpRouteGroup::Timed,
            path: "/{id}",
        },
        MachineOperation::Exec => HttpOperationBinding::Route {
            group: HttpRouteGroup::Timed,
            path: "/{id}/exec",
        },
        MachineOperation::ExecStream => HttpOperationBinding::Route {
            group: HttpRouteGroup::Timed,
            path: "/{id}/exec/stream",
        },
        MachineOperation::ExecInteractive => HttpOperationBinding::Route {
            group: HttpRouteGroup::LongLived,
            path: "/{id}/exec/interactive",
        },
        MachineOperation::Run => HttpOperationBinding::Route {
            group: HttpRouteGroup::Timed,
            path: "/{id}/run",
        },
        MachineOperation::RunSession => HttpOperationBinding::Route {
            group: HttpRouteGroup::Timed,
            path: "/run",
        },
        MachineOperation::Monitor => HttpOperationBinding::Route {
            group: HttpRouteGroup::LongLived,
            path: "/{id}/monitor",
        },
        MachineOperation::WriteFile => HttpOperationBinding::Route {
            group: HttpRouteGroup::Timed,
            path: "/{id}/files/{*path}",
        },
        MachineOperation::UploadFile => HttpOperationBinding::CoveredBy {
            operation: MachineOperation::WriteFile,
        },
        MachineOperation::ReadFile => HttpOperationBinding::Route {
            group: HttpRouteGroup::Timed,
            path: "/{id}/files/{*path}",
        },
        MachineOperation::DownloadFile => HttpOperationBinding::CoveredBy {
            operation: MachineOperation::ReadFile,
        },
        MachineOperation::StorageStatus => HttpOperationBinding::Route {
            group: HttpRouteGroup::Timed,
            path: "/{id}/storage",
        },
        MachineOperation::ListImages => HttpOperationBinding::Route {
            group: HttpRouteGroup::Timed,
            path: "/{id}/images",
        },
        MachineOperation::PullImage => HttpOperationBinding::Route {
            group: HttpRouteGroup::Timed,
            path: "/{id}/images/pull",
        },
        MachineOperation::PruneImages => HttpOperationBinding::Route {
            group: HttpRouteGroup::Timed,
            path: "/{id}/images/prune",
        },
        MachineOperation::NetworkTest => HttpOperationBinding::Route {
            group: HttpRouteGroup::Timed,
            path: "/{id}/network-test",
        },
        MachineOperation::DataDir => HttpOperationBinding::Route {
            group: HttpRouteGroup::Timed,
            path: "/{id}/data-dir",
        },
    }
}

fn machine_operation_method_router(
    operation: MachineOperation,
) -> Option<MethodRouter<Arc<ApiState>>> {
    match operation {
        MachineOperation::Create => Some(post(handlers::machines::create_machine)),
        MachineOperation::Status => Some(get(handlers::machines::get_machine)),
        MachineOperation::List => Some(get(handlers::machines::list_machines)),
        MachineOperation::Start => Some(post(handlers::machines::start_machine)),
        MachineOperation::Stop => Some(post(handlers::machines::stop_machine)),
        MachineOperation::Delete => Some(delete(handlers::machines::delete_machine)),
        MachineOperation::Fork => Some(post(handlers::machines::fork_machine)),
        MachineOperation::Update => Some(patch(handlers::machines::update_machine)),
        MachineOperation::Exec => Some(post(handlers::exec::exec_command)),
        MachineOperation::ExecStream => Some(post(handlers::exec::exec_stream)),
        MachineOperation::ExecInteractive => Some(get(handlers::exec::exec_interactive)),
        MachineOperation::Run => Some(post(handlers::exec::run_command)),
        MachineOperation::RunSession => Some(post(handlers::exec::run_session)),
        MachineOperation::Monitor => Some(get(handlers::machines::monitor_machine)),
        MachineOperation::WriteFile => Some(put(handlers::files::upload_file)),
        MachineOperation::UploadFile => None,
        MachineOperation::ReadFile => Some(get(handlers::files::download_file)),
        MachineOperation::DownloadFile => None,
        MachineOperation::StorageStatus => Some(get(handlers::images::storage_status)),
        MachineOperation::ListImages => Some(get(handlers::images::list_images)),
        MachineOperation::PullImage => Some(post(handlers::images::pull_image)),
        MachineOperation::PruneImages => Some(post(handlers::images::prune_images)),
        MachineOperation::NetworkTest => Some(post(handlers::machines::network_test)),
        MachineOperation::DataDir => Some(get(handlers::machines::data_dir)),
    }
}

fn machine_operation_routes(group: HttpRouteGroup) -> Router<Arc<ApiState>> {
    let mut router = Router::new();
    for operation in MachineOperation::ALL {
        match machine_operation_http_binding(*operation) {
            HttpOperationBinding::Route {
                group: route_group,
                path,
            } if route_group == group => {
                let method_router = machine_operation_method_router(*operation)
                    .expect("routed machine operation must have a method router");
                router = router.route(path, method_router);
            }
            HttpOperationBinding::Route { .. } => {}
            HttpOperationBinding::CoveredBy { operation } => {
                let _ = operation.method_name();
            }
        }
    }
    router
}

/// Create the API router with all endpoints.
///
/// `cors_origins` specifies allowed CORS origins. If empty, defaults to
/// localhost:8080 and localhost:3000 (both http and 127.0.0.1 variants).
pub fn create_router(state: Arc<ApiState>, cors_origins: Vec<String>) -> Router {
    // Health check route
    let health_route = Router::new().route("/health", get(handlers::health::health));

    // Node capacity introspection (polled by a fleet node-agent over HTTP).
    let capacity_route = Router::new().route("/capacity", get(handlers::node::capacity));

    // Long-lived streaming routes (no request timeout): logs, monitor SSE,
    // and the interactive PTY WebSocket can outlive the 5-minute API timeout.
    let logs_route = Router::new().route("/{id}/logs", get(handlers::exec::stream_logs));
    let long_lived_machine_routes = machine_operation_routes(HttpRouteGroup::LongLived);

    // Machine routes with timeout. Built from the machine operation catalog so
    // adding a service operation forces an HTTP route/coverage decision.
    let machine_routes_with_timeout =
        machine_operation_routes(HttpRouteGroup::Timed).layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(API_REQUEST_TIMEOUT_SECS),
        ));

    // Machine routes
    let machine_routes = Router::new()
        .merge(logs_route)
        .merge(long_lived_machine_routes)
        .merge(machine_routes_with_timeout);

    // API v1 routes
    let api_v1 = Router::new().nest("/machines", machine_routes);

    // CORS: Use configured origins, or default to localhost for security.
    let default_origins = || {
        vec![
            "http://localhost:8080"
                .parse()
                .expect("hardcoded CORS origin"),
            "http://127.0.0.1:8080"
                .parse()
                .expect("hardcoded CORS origin"),
            "http://localhost:3000"
                .parse()
                .expect("hardcoded CORS origin"),
            "http://127.0.0.1:3000"
                .parse()
                .expect("hardcoded CORS origin"),
        ]
    };
    let origins: Vec<axum::http::HeaderValue> = if cors_origins.is_empty() {
        default_origins()
    } else {
        let mut valid = Vec::new();
        for origin in &cors_origins {
            match origin.parse() {
                Ok(v) => valid.push(v),
                Err(e) => {
                    tracing::warn!(origin = %origin, error = %e, "invalid CORS origin, skipping");
                }
            }
        }
        if valid.is_empty() {
            tracing::warn!("no valid CORS origins provided, falling back to defaults");
            default_origins()
        } else {
            valid
        }
    };

    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::list(origins))
        .allow_methods([
            axum::http::Method::GET,
            axum::http::Method::POST,
            axum::http::Method::DELETE,
        ])
        .allow_headers([axum::http::header::CONTENT_TYPE]);

    // Prometheus metrics
    let metrics_route = Router::new().route("/metrics", get(serve_metrics));

    // Combine all routes
    Router::new()
        .merge(health_route)
        .merge(capacity_route)
        .merge(metrics_route)
        .nest("/api/v1", api_v1)
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .layer(middleware::from_fn(trace_id_middleware))
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .with_state(state)
}

/// Install the global Prometheus metrics recorder.
/// Returns None if a recorder is already installed (e.g., in tests).
pub fn install_metrics_recorder() -> Option<metrics_exporter_prometheus::PrometheusHandle> {
    metrics_exporter_prometheus::PrometheusBuilder::new()
        .install_recorder()
        .ok()
}

/// Serve Prometheus metrics as text.
async fn serve_metrics() -> String {
    METRICS_HANDLE.get().map(|h| h.render()).unwrap_or_default()
}

/// Global handle to the Prometheus recorder, set once at startup.
/// Only accessed by serve.rs (startup) and serve_metrics (handler).
pub static METRICS_HANDLE: std::sync::OnceLock<metrics_exporter_prometheus::PrometheusHandle> =
    std::sync::OnceLock::new();

/// Normalize a request path for Prometheus labels.
/// Replaces machine IDs with `:id` to prevent cardinality explosion.
fn normalize_metrics_path(path: &str) -> String {
    let parts: Vec<&str> = path.split('/').collect();
    if parts.len() >= 4 && parts[1] == "api" && parts[3] == "machines" {
        if let Some(id_pos) = parts.get(4) {
            if !id_pos.is_empty() {
                let mut normalized = parts[..4].to_vec();
                normalized.push(":id");
                normalized.extend_from_slice(&parts[5..]);
                return normalized.join("/");
            }
        }
    }
    path.to_string()
}

/// Trace ID for correlating API requests to agent operations.
#[derive(Clone, Debug)]
pub struct TraceId(pub String);

/// Middleware that generates a unique trace ID for each request and returns it
/// in the `X-Trace-Id` response header.
async fn trace_id_middleware(mut req: Request, next: Next) -> Response {
    use std::sync::atomic::{AtomicU64, Ordering};
    use tracing::Instrument;

    static REQUEST_SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = REQUEST_SEQ.fetch_add(1, Ordering::Relaxed);
    let trace_id = format!("{:08x}{:08x}", seq, std::process::id());

    req.extensions_mut().insert(TraceId(trace_id.clone()));

    let method = req.method().to_string();
    // Normalize path to template to avoid cardinality explosion from machine IDs
    let path_template = normalize_metrics_path(req.uri().path());

    let span = tracing::info_span!("request", trace_id = %trace_id);
    let mut response = next.run(req).instrument(span).await;

    let status = response.status().as_u16().to_string();
    metrics::counter!("smolvm_api_requests_total", "method" => method, "status" => status, "path" => path_template).increment(1);

    if let Ok(val) = HeaderValue::from_str(&trace_id) {
        response.headers_mut().insert("x-trace-id", val);
    }
    response
}

#[cfg(test)]
mod tests {
    use super::validate_command;

    #[test]
    fn test_validate_command() {
        assert!(validate_command(&[]).is_err());
        assert!(validate_command(&["echo".to_string()]).is_ok());
        assert!(validate_command(&["echo".to_string(), "hello".to_string()]).is_ok());
    }
}

#[cfg(test)]
mod operation_route_tests {
    use super::*;

    #[test]
    fn machine_operation_routes_build_for_all_groups() {
        let _ = machine_operation_routes(HttpRouteGroup::Timed);
        let _ = machine_operation_routes(HttpRouteGroup::LongLived);
    }

    #[test]
    fn covered_operations_reference_known_operations() {
        for operation in MachineOperation::ALL {
            if let HttpOperationBinding::CoveredBy {
                operation: covered_by,
            } = machine_operation_http_binding(*operation)
            {
                assert!(MachineOperation::ALL.contains(&covered_by));
            }
        }
    }
}
