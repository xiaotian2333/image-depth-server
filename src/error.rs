use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

#[derive(Debug)]
pub enum AppError {
    InvalidHash(String),
    DownloadFailed(String),
    DownloadTooLarge(String),
    ImageDecodeFailed(String),
    ModelMetadata(String),
    InferenceFailed(String),
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, msg) = match self {
            AppError::InvalidHash(m) => (StatusCode::BAD_REQUEST, m),
            AppError::DownloadFailed(m) => (StatusCode::BAD_GATEWAY, m),
            AppError::DownloadTooLarge(m) => (StatusCode::PAYLOAD_TOO_LARGE, m),
            AppError::ImageDecodeFailed(m) => (StatusCode::UNPROCESSABLE_ENTITY, m),
            AppError::ModelMetadata(m) => (StatusCode::INTERNAL_SERVER_ERROR, m),
            AppError::InferenceFailed(m) => (StatusCode::INTERNAL_SERVER_ERROR, m),
            AppError::Internal(m) => (StatusCode::INTERNAL_SERVER_ERROR, m),
        };

        (status, Json(json!({ "error": msg }))).into_response()
    }
}
