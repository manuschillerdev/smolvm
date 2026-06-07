//! File I/O handlers — upload and download files to/from a running machine.

use axum::{
    body::Bytes,
    extract::{Path, State},
    Json,
};
use serde::Serialize;
use std::sync::Arc;
use utoipa::ToSchema;

use crate::api::error::ApiError;
use crate::api::state::ApiState;
use crate::api::TraceId;
use crate::machine::{LocalMachineService, MachineService, ReadMachineFile, WriteMachineFile};

/// Response from file upload.
#[derive(Debug, Serialize, ToSchema)]
pub struct FileUploadResponse {
    /// Path where the file was written.
    pub path: String,
    /// Size of the file in bytes.
    pub size: u64,
}

/// Upload a file to a machine.
///
/// Writes the request body as a file at the specified path inside the VM.
/// Creates parent directories automatically.
#[utoipa::path(
    put,
    path = "/api/v1/machines/{id}/files/{path}",
    tag = "Files",
    params(
        ("id" = String, Path, description = "Machine name"),
        ("path" = String, Path, description = "File path inside the VM (e.g., workspace/script.py)")
    ),
    request_body(content = Vec<u8>, content_type = "application/octet-stream"),
    responses(
        (status = 200, description = "File uploaded", body = FileUploadResponse),
        (status = 404, description = "Machine not found"),
        (status = 500, description = "Write failed")
    )
)]
pub async fn upload_file(
    State(state): State<Arc<ApiState>>,
    Path((id, file_path)): Path<(String, String)>,
    trace_id: Option<axum::Extension<TraceId>>,
    body: Bytes,
) -> Result<Json<FileUploadResponse>, ApiError> {
    let file_path = file_path.trim_start_matches('/');
    let guest_path = format!("/{}", file_path);
    let size = body.len() as u64;
    let request = WriteMachineFile {
        name: id,
        guest_path: guest_path.clone(),
        data: body.to_vec(),
        mode: None,
        start_if_needed: true,
        trace_id: trace_id.map(|t| t.0 .0.clone()),
    };

    let db = state.db().clone();
    tokio::task::spawn_blocking(move || LocalMachineService::with_db(db).write_file(request))
        .await
        .map_err(|e| ApiError::internal(format!("task error: {}", e)))?
        .map_err(ApiError::from)?;

    Ok(Json(FileUploadResponse {
        path: guest_path,
        size,
    }))
}

/// Download a file from a machine.
///
/// Returns the file contents as a raw byte stream.
#[utoipa::path(
    get,
    path = "/api/v1/machines/{id}/files/{path}",
    tag = "Files",
    params(
        ("id" = String, Path, description = "Machine name"),
        ("path" = String, Path, description = "File path inside the VM")
    ),
    responses(
        (status = 200, description = "File contents", content_type = "application/octet-stream"),
        (status = 404, description = "Machine or file not found"),
        (status = 500, description = "Read failed")
    )
)]
pub async fn download_file(
    State(state): State<Arc<ApiState>>,
    Path((id, file_path)): Path<(String, String)>,
    trace_id: Option<axum::Extension<TraceId>>,
) -> Result<Bytes, ApiError> {
    let file_path = file_path.trim_start_matches('/');
    let request = ReadMachineFile {
        name: id,
        guest_path: format!("/{}", file_path),
        start_if_needed: true,
        trace_id: trace_id.map(|t| t.0 .0.clone()),
    };

    let db = state.db().clone();
    let data =
        tokio::task::spawn_blocking(move || LocalMachineService::with_db(db).read_file(request))
            .await
            .map_err(|e| ApiError::internal(format!("task error: {}", e)))?
            .map_err(ApiError::from)?;

    Ok(Bytes::from(data))
}
