use serde::{Deserialize, Serialize};
use sled::Db;

use crate::error::AppError;

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct DepthEntry {
    pub hash: String,
    pub width: u32,
    pub height: u32,
    pub data_url: String,
    pub created_at: i64,
}

pub struct DepthCache {
    db: Db,
}

impl DepthCache {
    pub fn new(path: &str) -> Result<Self, AppError> {
        let db = sled::open(path).map_err(|e| AppError::Internal(format!("sled 打开失败: {e}")))?;
        Ok(Self { db })
    }

    pub fn make_key(
        hash: &str,
        model_fingerprint: &str,
        cover_size: u32,
        max_image_size: u32,
    ) -> String {
        format!(
            "v2:{model_fingerprint}:cover-{cover_size}:max-{max_image_size}:{hash}"
        )
    }

    pub fn get(&self, key: &str) -> Result<Option<DepthEntry>, AppError> {
        match self.db.get(key) {
            Ok(Some(bytes)) => {
                let entry = serde_json::from_slice(&bytes)
                    .map_err(|e| AppError::Internal(format!("缓存反序列化失败: {e}")))?;
                Ok(Some(entry))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(AppError::Internal(format!("sled 读取失败: {e}"))),
        }
    }

    pub fn set(&self, key: &str, entry: &DepthEntry) -> Result<(), AppError> {
        let bytes = serde_json::to_vec(entry)
            .map_err(|e| AppError::Internal(format!("缓存序列化失败: {e}")))?;

        self.db
            .insert(key.as_bytes(), bytes)
            .map_err(|e| AppError::Internal(format!("sled 写入失败: {e}")))?;
        self.db
            .flush()
            .map_err(|e| AppError::Internal(format!("sled 刷盘失败: {e}")))?;

        Ok(())
    }

    pub fn len(&self) -> usize {
        self.db.len()
    }
}
