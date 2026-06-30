use std::sync::Arc;

use axum::{
    extract::{Path, State},
    Json,
};
use serde_json::{json, Value};
use tokio::sync::Semaphore;

use crate::{
    cache::{DepthCache, DepthEntry},
    config::Config,
    download::CoverDownloader,
    error::AppError,
    inference::DepthModel,
    postprocess::depth_to_data_url,
    preprocess::preprocess,
};

pub struct AppState {
    pub config: Config,
    pub model: Arc<DepthModel>,
    pub downloader: CoverDownloader,
    pub cache: DepthCache,
    pub infer_sem: Arc<Semaphore>,
}

pub async fn get_depth(
    Path(hash): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<DepthEntry>, AppError> {
    if hash.len() != state.config.hash_digits || !hash.chars().all(|c| c.is_ascii_digit()) {
        return Err(AppError::InvalidHash(format!(
            "hash 必须是 {} 位数字",
            state.config.hash_digits
        )));
    }

    let cache_key = DepthCache::make_key(
        &hash,
        state.model.cache_fingerprint(),
        state.config.cover_size,
        state.config.max_image_size,
    );
    if let Some(entry) = state.cache.get(&cache_key)? {
        return Ok(Json(entry));
    }

    let img_bytes = state
        .downloader
        .download_cover(&hash, state.config.max_download_bytes)
        .await?;

    let prepared = preprocess(
        &img_bytes,
        state.model.input_width(),
        state.model.input_height(),
        state.config.max_image_size,
    )?;

    let permit = state
        .infer_sem
        .clone()
        .acquire_owned()
        .await
        .map_err(|e| AppError::Internal(format!("推理信号量关闭: {e}")))?;
    let depth = state.model.clone().run_blocking(prepared.tensor).await?;
    drop(permit);

    let data_url = depth_to_data_url(&depth, &prepared.meta)?;
    let entry = DepthEntry {
        hash: hash.clone(),
        width: prepared.meta.output_width,
        height: prepared.meta.output_height,
        data_url,
        created_at: chrono::Utc::now().timestamp_millis(),
    };

    state.cache.set(&cache_key, &entry)?;

    Ok(Json(entry))
}

pub async fn health(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "cacheSize": state.cache.len(),
        "modelInput": {
            "width": state.model.input_width(),
            "height": state.model.input_height()
        }
    }))
}
