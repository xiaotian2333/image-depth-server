use clap::Parser;

use crate::error::AppError;

#[derive(Parser, Debug, Clone)]
#[command(name = "depth-server")]
pub struct Config {
    #[arg(long, env = "DEPTH_PORT", default_value_t = 7860)]
    pub port: u16,

    #[arg(
        long = "model",
        env = "DEPTH_MODEL_PATH",
        default_value = "./model_quantized.onnx"
    )]
    pub model_path: String,

    #[arg(long, env = "DEPTH_CACHE_DIR", default_value = "./cache")]
    pub cache_dir: String,

    #[arg(long, env = "DEPTH_DOWNLOAD_TIMEOUT_SECS", default_value_t = 10)]
    pub download_timeout_secs: u64,

    #[arg(long, env = "DEPTH_MAX_DOWNLOAD_BYTES", default_value_t = 5 * 1024 * 1024)]
    pub max_download_bytes: usize,

    #[arg(long, env = "DEPTH_MAX_IMAGE_SIZE", default_value_t = 1024)]
    pub max_image_size: u32,

    #[arg(long, env = "DEPTH_TARGET_SIZE", default_value_t = 518)]
    pub target_size: u32,

    #[arg(long, env = "DEPTH_HASH_DIGITS", default_value_t = 20)]
    pub hash_digits: usize,

    #[arg(long, env = "DEPTH_INFER_CONCURRENCY", default_value_t = 1)]
    pub infer_concurrency: usize,
}

impl Config {
    pub fn from_env() -> Self {
        Self::parse()
    }

    pub fn validate(self) -> Result<Self, AppError> {
        if self.download_timeout_secs == 0 {
            return Err(AppError::Internal("下载超时时间必须大于 0".into()));
        }
        if self.max_download_bytes == 0 {
            return Err(AppError::Internal("下载大小限制必须大于 0".into()));
        }
        if self.max_image_size == 0 {
            return Err(AppError::Internal("最大图片尺寸必须大于 0".into()));
        }
        if self.target_size == 0 {
            return Err(AppError::Internal("推理目标尺寸必须大于 0".into()));
        }
        if self.hash_digits < 8 {
            return Err(AppError::Internal("hash 位数必须至少为 8".into()));
        }

        Ok(self)
    }
}
