mod cache;
mod config;
mod download;
mod error;
mod inference;
mod postprocess;
mod preprocess;
mod routes;

use std::{net::SocketAddr, sync::Arc};

use axum::{routing::get, Router};
use cache::DepthCache;
use config::Config;
use download::CoverDownloader;
use error::AppError;
use inference::DepthModel;
use routes::{get_depth, health, AppState};
use tokio::sync::Semaphore;
use tower_http::{cors::CorsLayer, trace::TraceLayer};

#[tokio::main]
async fn main() -> Result<(), AppError> {
    tracing_subscriber::fmt()
        .with_env_filter("image_depth_server=info,tower_http=info")
        .init();

    let config = Config::from_env().validate()?;
    let port = config.port;
    let infer_concurrency = config.infer_concurrency.max(1);

    let model = Arc::new(DepthModel::new(&config.model_path, config.target_size)?);
    let downloader = CoverDownloader::new(config.download_timeout_secs)?;
    let cache = DepthCache::new(&config.cache_dir)?;

    let state = Arc::new(AppState {
        config,
        model,
        downloader,
        cache,
        infer_sem: Arc::new(Semaphore::new(infer_concurrency)),
    });

    let app = Router::new()
        .route("/depth/{hash}", get(get_depth))
        .route("/health", get(health))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("image-depth-server 启动于 {addr}");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| AppError::Internal(format!("端口绑定失败: {e}")))?;

    axum::serve(listener, app)
        .await
        .map_err(|e| AppError::Internal(format!("HTTP 服务失败: {e}")))?;

    Ok(())
}
