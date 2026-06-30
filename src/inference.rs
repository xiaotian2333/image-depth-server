use std::{
    path::Path,
    sync::{Arc, Mutex},
    time::UNIX_EPOCH,
};

use ndarray::{Array2, Array4};
use ort::{
    session::Session,
    value::{Outlet, Tensor, TensorElementType, ValueType},
};

use crate::error::AppError;

pub struct DepthModel {
    session: Mutex<Session>,
    input_name: String,
    output_name: String,
    input_width: u32,
    input_height: u32,
    cache_fingerprint: String,
}

impl DepthModel {
    pub fn new(model_path: &str, fallback_target_size: u32) -> Result<Self, AppError> {
        let session = Session::builder()
            .map_err(|e| AppError::Internal(format!("创建 ONNX session builder 失败: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| AppError::Internal(format!("模型加载失败: {e}")))?;

        let input = session
            .inputs()
            .first()
            .ok_or_else(|| AppError::ModelMetadata("模型没有输入".into()))?;
        let output = session
            .outputs()
            .first()
            .ok_or_else(|| AppError::ModelMetadata("模型没有输出".into()))?;

        let input_name = input.name().to_string();
        let output_name = output.name().to_string();
        let (input_width, input_height) = resolve_nchw_input_size(input, fallback_target_size)?;
        validate_depth_output(output)?;

        tracing::info!(
            input = %input_name,
            output = %output_name,
            input_width,
            input_height,
            input_type = ?input.dtype(),
            output_type = ?output.dtype(),
            "ONNX 模型元信息"
        );

        Ok(Self {
            session: Mutex::new(session),
            input_name,
            output_name,
            input_width,
            input_height,
            cache_fingerprint: build_model_fingerprint(model_path, input_width, input_height)?,
        })
    }

    pub fn input_width(&self) -> u32 {
        self.input_width
    }

    pub fn input_height(&self) -> u32 {
        self.input_height
    }

    pub fn cache_fingerprint(&self) -> &str {
        &self.cache_fingerprint
    }

    pub async fn run_blocking(
        self: Arc<Self>,
        input: Array4<f32>,
    ) -> Result<Array2<f32>, AppError> {
        tokio::task::spawn_blocking(move || self.run_inner(input))
            .await
            .map_err(|e| AppError::InferenceFailed(format!("推理任务 join 失败: {e}")))?
    }

    fn run_inner(&self, input: Array4<f32>) -> Result<Array2<f32>, AppError> {
        let mut session = self
            .session
            .lock()
            .map_err(|_| AppError::InferenceFailed("ONNX session 锁已损坏".into()))?;

        let tensor = Tensor::from_array(input)
            .map_err(|e| AppError::InferenceFailed(format!("输入张量创建失败: {e}")))?;

        let outputs = session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(|e| AppError::InferenceFailed(format!("推理失败: {e}")))?;

        let output = outputs
            .get(self.output_name.as_str())
            .ok_or_else(|| AppError::InferenceFailed("模型输出缺少预期名称".into()))?;
        let (shape, data) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| AppError::InferenceFailed(format!("输出提取失败: {e}")))?;

        squeeze_depth_to_2d(shape, data)
    }
}

fn resolve_nchw_input_size(
    input: &Outlet,
    fallback_target_size: u32,
) -> Result<(u32, u32), AppError> {
    match input.dtype() {
        ValueType::Tensor { ty, shape, .. } => {
            if *ty != TensorElementType::Float32 {
                return Err(AppError::ModelMetadata(format!(
                    "模型输入必须是 f32 tensor，当前为 {ty}"
                )));
            }
            if shape.len() != 4 {
                return Err(AppError::ModelMetadata(format!(
                    "模型输入必须是 NCHW 4 维张量，当前 shape 为 {shape}"
                )));
            }
            if shape[1] > 0 && shape[1] != 3 {
                return Err(AppError::ModelMetadata(format!(
                    "模型输入通道数必须为 3，当前 shape 为 {shape}"
                )));
            }

            let height = resolve_model_dim(shape[2], fallback_target_size, "输入高度")?;
            let width = resolve_model_dim(shape[3], fallback_target_size, "输入宽度")?;
            Ok((width, height))
        }
        other => Err(AppError::ModelMetadata(format!(
            "模型输入必须是 tensor，当前为 {other}"
        ))),
    }
}

fn validate_depth_output(output: &Outlet) -> Result<(), AppError> {
    match output.dtype() {
        ValueType::Tensor { ty, shape, .. } => {
            if *ty != TensorElementType::Float32 {
                return Err(AppError::ModelMetadata(format!(
                    "模型输出必须是 f32 tensor，当前为 {ty}"
                )));
            }
            if !(2..=4).contains(&shape.len()) {
                return Err(AppError::ModelMetadata(format!(
                    "模型输出必须是 2 到 4 维深度张量，当前 shape 为 {shape}"
                )));
            }
            Ok(())
        }
        other => Err(AppError::ModelMetadata(format!(
            "模型输出必须是 tensor，当前为 {other}"
        ))),
    }
}

fn resolve_model_dim(dim: i64, fallback: u32, label: &str) -> Result<u32, AppError> {
    if dim > 0 {
        u32::try_from(dim).map_err(|_| AppError::ModelMetadata(format!("{label} 超出 u32 范围")))
    } else if fallback > 0 {
        Ok(fallback)
    } else {
        Err(AppError::ModelMetadata(format!(
            "{label} 是动态维度，但 fallback target_size 无效"
        )))
    }
}

fn squeeze_depth_to_2d(shape: &[i64], data: &[f32]) -> Result<Array2<f32>, AppError> {
    let dims = shape
        .iter()
        .map(|dim| {
            usize::try_from(*dim)
                .map_err(|_| AppError::InferenceFailed(format!("深度输出包含非法维度: {shape:?}")))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let (height, width) = match dims.as_slice() {
        [h, w] => (*h, *w),
        [1, h, w] => (*h, *w),
        [1, 1, h, w] => (*h, *w),
        _ => {
            return Err(AppError::InferenceFailed(format!(
                "不支持的深度输出 shape: {shape:?}"
            )))
        }
    };

    if height == 0 || width == 0 {
        return Err(AppError::InferenceFailed("深度输出尺寸为空".into()));
    }
    if data.len() != height * width {
        return Err(AppError::InferenceFailed(format!(
            "深度输出元素数量不匹配，shape={shape:?}, len={}",
            data.len()
        )));
    }

    Array2::from_shape_vec((height, width), data.to_vec())
        .map_err(|e| AppError::InferenceFailed(format!("深度输出转换失败: {e}")))
}

fn build_model_fingerprint(
    model_path: &str,
    input_width: u32,
    input_height: u32,
) -> Result<String, AppError> {
    let path = Path::new(model_path);
    let metadata = path
        .metadata()
        .map_err(|e| AppError::Internal(format!("读取模型文件信息失败: {e}")))?;
    let modified_ms = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis())
        .unwrap_or_default();

    let raw = format!(
        "{model_path}\0{}\0{modified_ms}\0{input_width}\0{input_height}\0preprocess-v1",
        metadata.len()
    );

    Ok(format!("{:016x}", fnv1a64(raw.as_bytes())))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
