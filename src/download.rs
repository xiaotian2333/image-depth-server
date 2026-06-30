use std::time::Duration;

use futures_util::StreamExt;

use crate::error::AppError;

pub struct CoverDownloader {
    client: reqwest::Client,
    cover_size: u32,
}

impl CoverDownloader {
    pub fn new(timeout_secs: u64, cover_size: u32) -> Result<Self, AppError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .user_agent("image-depth-server/0.1")
            .build()
            .map_err(|e| AppError::Internal(format!("HTTP 客户端创建失败: {e}")))?;

        Ok(Self { client, cover_size })
    }

    pub async fn download_cover(&self, hash: &str, max_bytes: usize) -> Result<Vec<u8>, AppError> {
        let prefix = &hash[..8];
        let url = format!(
            "https://imge.kugou.com/stdmusic/{}/{prefix}/{hash}.jpg",
            self.cover_size
        );

        let resp = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| AppError::DownloadFailed(format!("下载失败: {e}")))?;

        if !resp.status().is_success() {
            return Err(AppError::DownloadFailed(format!(
                "CDN 返回 HTTP {}",
                resp.status()
            )));
        }

        if resp
            .content_length()
            .is_some_and(|len| len as usize > max_bytes)
        {
            return Err(AppError::DownloadTooLarge("图片响应超过大小限制".into()));
        }

        let mut out = Vec::new();
        let mut stream = resp.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk =
                chunk.map_err(|e| AppError::DownloadFailed(format!("读取响应失败: {e}")))?;
            out.extend_from_slice(&chunk);

            if out.len() > max_bytes {
                return Err(AppError::DownloadTooLarge("图片响应超过大小限制".into()));
            }
        }

        Ok(out)
    }
}
