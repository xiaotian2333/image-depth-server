use image::{imageops, Rgb, RgbImage};
use ndarray::Array4;

use crate::error::AppError;

pub struct PreprocessOutput {
    pub tensor: Array4<f32>,
    pub meta: PreprocessMeta,
}

pub struct PreprocessMeta {
    pub output_width: u32,
    pub output_height: u32,
    pub target_width: u32,
    pub target_height: u32,
    pub infer_width: u32,
    pub infer_height: u32,
}

pub fn preprocess(
    bytes: &[u8],
    target_width: u32,
    target_height: u32,
    max_image_size: u32,
) -> Result<PreprocessOutput, AppError> {
    if target_width == 0 || target_height == 0 || max_image_size == 0 {
        return Err(AppError::Internal("图片处理尺寸必须大于 0".into()));
    }

    let img = image::load_from_memory(bytes)
        .map_err(|e| AppError::ImageDecodeFailed(format!("图片解码失败: {e}")))?
        .to_rgb8();

    let (orig_w, orig_h) = img.dimensions();
    if orig_w == 0 || orig_h == 0 {
        return Err(AppError::ImageDecodeFailed("图片尺寸为空".into()));
    }

    let limit_scale = (max_image_size as f64 / orig_w.max(orig_h) as f64).min(1.0);
    let output_width = ((orig_w as f64 * limit_scale).round() as u32).max(1);
    let output_height = ((orig_h as f64 * limit_scale).round() as u32).max(1);

    let work = if output_width == orig_w && output_height == orig_h {
        img
    } else {
        imageops::resize(
            &img,
            output_width,
            output_height,
            imageops::FilterType::Lanczos3,
        )
    };

    let infer_scale = f64::min(
        target_width as f64 / output_width as f64,
        target_height as f64 / output_height as f64,
    );
    let infer_width = ((output_width as f64 * infer_scale).round() as u32).clamp(1, target_width);
    let infer_height =
        ((output_height as f64 * infer_scale).round() as u32).clamp(1, target_height);

    let resized = imageops::resize(
        &work,
        infer_width,
        infer_height,
        imageops::FilterType::Lanczos3,
    );
    let mut padded = RgbImage::from_pixel(target_width, target_height, Rgb([0, 0, 0]));
    imageops::replace(&mut padded, &resized, 0, 0);

    let mean = [0.485f32, 0.456f32, 0.406f32];
    let std = [0.229f32, 0.224f32, 0.225f32];
    let mut tensor = Array4::zeros((1, 3, target_height as usize, target_width as usize));

    for y in 0..target_height {
        for x in 0..target_width {
            let p = padded.get_pixel(x, y);
            tensor[[0, 0, y as usize, x as usize]] = (p[0] as f32 / 255.0 - mean[0]) / std[0];
            tensor[[0, 1, y as usize, x as usize]] = (p[1] as f32 / 255.0 - mean[1]) / std[1];
            tensor[[0, 2, y as usize, x as usize]] = (p[2] as f32 / 255.0 - mean[2]) / std[2];
        }
    }

    Ok(PreprocessOutput {
        tensor,
        meta: PreprocessMeta {
            output_width,
            output_height,
            target_width,
            target_height,
            infer_width,
            infer_height,
        },
    })
}
