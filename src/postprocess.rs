use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use image::codecs::png::PngEncoder;
use image::{imageops, ColorType, GrayImage, ImageEncoder, Luma};
use ndarray::{s, Array2};

use crate::{error::AppError, preprocess::PreprocessMeta};

pub fn depth_to_data_url(depth: &Array2<f32>, meta: &PreprocessMeta) -> Result<String, AppError> {
    let (depth_h, depth_w) = depth.dim();
    if depth_h == 0 || depth_w == 0 {
        return Err(AppError::InferenceFailed("深度输出尺寸为空".into()));
    }

    let crop_w = ((meta.infer_width as f64 / meta.target_width as f64) * depth_w as f64)
        .round()
        .clamp(1.0, depth_w as f64) as usize;
    let crop_h = ((meta.infer_height as f64 / meta.target_height as f64) * depth_h as f64)
        .round()
        .clamp(1.0, depth_h as f64) as usize;

    let cropped = depth.slice(s![0..crop_h, 0..crop_w]);

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for v in cropped.iter().copied().filter(|v| v.is_finite()) {
        min = min.min(v);
        max = max.max(v);
    }

    if !min.is_finite() || !max.is_finite() {
        return Err(AppError::InferenceFailed("深度输出不包含有效数值".into()));
    }

    let range = (max - min).max(1e-6);
    let mut gray = GrayImage::new(crop_w as u32, crop_h as u32);

    for y in 0..crop_h {
        for x in 0..crop_w {
            let v = cropped[[y, x]];
            let px = if v.is_finite() {
                ((v - min) / range * 255.0).clamp(0.0, 255.0) as u8
            } else {
                0
            };
            gray.put_pixel(x as u32, y as u32, Luma([px]));
        }
    }

    let out = if gray.width() == meta.output_width && gray.height() == meta.output_height {
        gray
    } else {
        imageops::resize(
            &gray,
            meta.output_width,
            meta.output_height,
            imageops::FilterType::CatmullRom,
        )
    };

    let mut png_bytes = Vec::new();
    let encoder = PngEncoder::new(&mut png_bytes);
    encoder
        .write_image(
            out.as_raw(),
            out.width(),
            out.height(),
            ColorType::L8.into(),
        )
        .map_err(|e| AppError::Internal(format!("PNG 编码失败: {e}")))?;

    Ok(format!(
        "data:image/png;base64,{}",
        BASE64.encode(&png_bytes)
    ))
}
