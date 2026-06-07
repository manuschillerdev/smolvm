//! Image management handlers.

use axum::{
    extract::{Path, State},
    Json,
};
use std::sync::Arc;

use crate::api::error::ApiError;
use crate::api::state::ApiState;
use crate::api::types::{
    ApiErrorResponse, ImageInfo, ListImagesResponse, PruneImagesRequest, PruneImagesResponse,
    PullImageRequest, PullImageResponse, StorageStatusResponse,
};
use crate::api::TraceId;
use crate::machine::{
    ListMachineImages, LocalMachineService, MachineService, PruneMachineImages, PullMachineImage,
    StorageStatusRequest,
};

/// List images in a machine.
#[utoipa::path(
    get,
    path = "/api/v1/machines/{id}/images",
    tag = "Images",
    params(
        ("id" = String, Path, description = "Machine name")
    ),
    responses(
        (status = 200, description = "List of images", body = ListImagesResponse),
        (status = 404, description = "Machine not found", body = ApiErrorResponse)
    )
)]
pub async fn list_images(
    State(state): State<Arc<ApiState>>,
    Path(machine_id): Path<String>,
    trace_id: Option<axum::Extension<TraceId>>,
) -> Result<Json<ListImagesResponse>, ApiError> {
    let request = ListMachineImages {
        name: machine_id,
        start_if_needed: false,
        stop_after_start: false,
        empty_when_stopped: true,
        trace_id: trace_id.map(|t| t.0 .0.clone()),
    };
    let db = state.db().clone();
    let images =
        tokio::task::spawn_blocking(move || LocalMachineService::with_db(db).list_images(request))
            .await
            .map_err(|e| ApiError::internal(format!("task error: {}", e)))?
            .map_err(ApiError::from)?;

    let images = images
        .into_iter()
        .map(|i| ImageInfo {
            reference: i.reference,
            digest: i.digest,
            size: i.size,
            architecture: i.architecture,
            os: i.os,
            layer_count: i.layer_count,
        })
        .collect();

    Ok(Json(ListImagesResponse { images }))
}

/// Pull an image into a machine.
#[utoipa::path(
    post,
    path = "/api/v1/machines/{id}/images/pull",
    tag = "Images",
    params(
        ("id" = String, Path, description = "Machine name")
    ),
    request_body = PullImageRequest,
    responses(
        (status = 200, description = "Image pulled", body = PullImageResponse),
        (status = 400, description = "Invalid request", body = ApiErrorResponse),
        (status = 404, description = "Machine not found", body = ApiErrorResponse),
        (status = 500, description = "Failed to pull image", body = ApiErrorResponse)
    )
)]
pub async fn pull_image(
    State(state): State<Arc<ApiState>>,
    Path(machine_id): Path<String>,
    trace_id: Option<axum::Extension<TraceId>>,
    Json(req): Json<PullImageRequest>,
) -> Result<Json<PullImageResponse>, ApiError> {
    if req.image.is_empty() {
        return Err(ApiError::BadRequest(
            "image reference cannot be empty".into(),
        ));
    }

    let request = PullMachineImage {
        name: machine_id,
        image: req.image.clone(),
        oci_platform: req.oci_platform.clone(),
        proxy: req.proxy.clone(),
        no_proxy: req.no_proxy.clone(),
        start_if_needed: true,
        trace_id: trace_id.map(|t| t.0 .0.clone()),
    };

    let db = state.db().clone();
    let start = std::time::Instant::now();
    let image_info =
        tokio::task::spawn_blocking(move || LocalMachineService::with_db(db).pull_image(request))
            .await
            .map_err(|e| ApiError::internal(format!("task error: {}", e)))?
            .map_err(ApiError::from)?;
    metrics::histogram!("smolvm_image_pull_seconds").record(start.elapsed().as_secs_f64());

    Ok(Json(PullImageResponse {
        image: ImageInfo {
            reference: image_info.reference,
            digest: image_info.digest,
            size: image_info.size,
            architecture: image_info.architecture,
            os: image_info.os,
            layer_count: image_info.layer_count,
        },
    }))
}

/// Return OCI storage status for a machine.
#[utoipa::path(
    get,
    path = "/api/v1/machines/{id}/storage",
    tag = "Images",
    params(
        ("id" = String, Path, description = "Machine name")
    ),
    responses(
        (status = 200, description = "Storage status", body = StorageStatusResponse),
        (status = 404, description = "Machine not found", body = ApiErrorResponse),
        (status = 409, description = "Machine is not running", body = ApiErrorResponse)
    )
)]
pub async fn storage_status(
    State(state): State<Arc<ApiState>>,
    Path(machine_id): Path<String>,
    trace_id: Option<axum::Extension<TraceId>>,
) -> Result<Json<StorageStatusResponse>, ApiError> {
    let mut request = StorageStatusRequest::new(machine_id);
    request.start_if_needed = true;
    request.trace_id = trace_id.map(|t| t.0 .0.clone());

    let db = state.db().clone();
    let status = tokio::task::spawn_blocking(move || {
        LocalMachineService::with_db(db).storage_status(request)
    })
    .await
    .map_err(|e| ApiError::internal(format!("task error: {}", e)))?
    .map_err(ApiError::from)?;

    Ok(Json(StorageStatusResponse {
        ready: status.ready,
        total_bytes: status.total_bytes,
        used_bytes: status.used_bytes,
        image_count: status.image_count,
        layer_count: status.layer_count,
    }))
}

/// Prune image/layer storage for a machine.
#[utoipa::path(
    post,
    path = "/api/v1/machines/{id}/images/prune",
    tag = "Images",
    params(
        ("id" = String, Path, description = "Machine name")
    ),
    request_body = PruneImagesRequest,
    responses(
        (status = 200, description = "Images pruned", body = PruneImagesResponse),
        (status = 400, description = "Invalid request", body = ApiErrorResponse),
        (status = 404, description = "Machine not found", body = ApiErrorResponse),
        (status = 409, description = "Invalid state", body = ApiErrorResponse)
    )
)]
pub async fn prune_images(
    State(state): State<Arc<ApiState>>,
    Path(machine_id): Path<String>,
    trace_id: Option<axum::Extension<TraceId>>,
    Json(req): Json<PruneImagesRequest>,
) -> Result<Json<PruneImagesResponse>, ApiError> {
    let mut request = PruneMachineImages::new(machine_id);
    request.dry_run = req.dry_run;
    request.all = req.all;
    request.stop_after_start = req.stop_after_start;
    request.trace_id = trace_id.map(|t| t.0 .0.clone());

    let db = state.db().clone();
    let result =
        tokio::task::spawn_blocking(move || LocalMachineService::with_db(db).prune_images(request))
            .await
            .map_err(|e| ApiError::internal(format!("task error: {}", e)))?
            .map_err(ApiError::from)?;

    Ok(Json(PruneImagesResponse {
        freed_bytes: result.freed_bytes,
        removed_images: result.removed_images,
        dry_run: result.dry_run,
    }))
}
