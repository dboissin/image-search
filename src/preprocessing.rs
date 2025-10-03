use std::{ffi::OsStr, path::Path};

use candle_core::{DType, Device, Tensor};
use image::imageops::FilterType;
use opencv::{boxed_ref::BoxedRef, core::{bitwise_not, find_non_zero, no_array, Mat, MatTraitConst, Point, Point2f, Scalar, Size, Vector, BORDER_CONSTANT, BORDER_DEFAULT, BORDER_REPLICATE}, imgcodecs::{imread, imwrite, IMREAD_COLOR}, imgproc::{bounding_rect, cvt_color, dilate, find_contours, gaussian_blur, get_rotation_matrix_2d, get_structuring_element, min_area_rect, morphology_default_border_value, resize, threshold, warp_affine, CHAIN_APPROX_SIMPLE, COLOR_BGR2GRAY, INTER_AREA, INTER_CUBIC, MORPH_RECT, RETR_EXTERNAL, THRESH_BINARY, THRESH_OTSU}, Error};
use tempfile::tempdir;

pub(crate) struct ImageTensorConfig {
    pad: bool,
    shape: (usize, usize, usize),
    normalization_mean: Vec<f32>,
    normalization_std_dev: Vec<f32>,
    resample: u8,
    rescale_factor: Option<f64>,
    max_width: i32,
    max_height: i32,
}

impl Default for ImageTensorConfig {
    fn default() -> Self {
        Self {
            pad: false,
            shape: (384, 384, 3),
            normalization_mean: vec![0.5, 0.5, 0.5],
            normalization_std_dev: vec![0.5, 0.5, 0.5],
            resample: 2,
            rescale_factor: Some(1.0/255.0),
            max_width: 1024,
            max_height: 768
        }
    }
}

fn resample_filter_type(resample: u8) -> FilterType {
    match resample {
        0 => FilterType::Nearest,       // Pillow NEAREST
        1 => FilterType::Lanczos3,      // Pillow LANCZOS (or ANTIALIAS)
        2 => FilterType::Triangle,      // Pillow BILINEAR
        3 => FilterType::CatmullRom,    // Pillow BICUBIC
        _ => {
            FilterType::CatmullRom
        }
    }
}

pub(crate) fn load_image<P: AsRef<Path>>(p: P, config: &ImageTensorConfig, device: &Device) -> crate::Result<Tensor> {
    let img = image::ImageReader::open(p)?.decode()?;
    let img = if config.pad {
        img.resize_to_fill(config.shape.0 as u32, config.shape.1 as u32, resample_filter_type(config.resample))
    } else {
        img.resize_exact(config.shape.0 as u32, config.shape.1 as u32, resample_filter_type(config.resample))
    };

    let mut data = Tensor::from_vec(img.to_rgb8().into_raw(), config.shape, device)?.permute((2, 0, 1))?.to_dtype(DType::F32)?;
    let mean = Tensor::new(config.normalization_mean.as_slice(), device)?.reshape((3, 1, 1))?;
    let std = Tensor::new(config.normalization_std_dev.as_slice(), device)?.reshape((3, 1, 1))?;

    if let Some(rescale_factor) = config.rescale_factor {
        data = (data * rescale_factor)?;
    }
    Ok(data.broadcast_sub(&mean)?.broadcast_div(&std)?)
}

fn resize_if_needed<P: AsRef<Path>>(p: P, max_width: i32, max_height: i32) -> Result<Mat, Error> {
    let img = imread(&format!("{}", p.as_ref().display()), IMREAD_COLOR)?;
    let size = img.size()?;
    if size.width > max_width || size.height > max_height {
        let scale_w = max_width as f64 / size.width as f64;
        let scale_h = max_height as f64 / size.height as f64;
        let scale = scale_w.min(scale_h);
        let width = (size.width as f64 * scale).round() as i32;
        let height = (size.height as f64 * scale).round() as i32;
        let mut resized = Mat::default();
        resize(&img, &mut resized, Size::new(width, height), 0.0, 0.0, INTER_AREA)?;
        Ok(resized)
    } else {
        Ok(img)
    }
}

fn binarize_inv(img: &Mat, blur: Option<Size>) -> Result<Mat, Error> {
    let mut gray = Mat::default();
    cvt_color(&img, &mut gray, COLOR_BGR2GRAY, 0)?;
    let blurred_or_not = if let Some(blur_size) = blur {
        let mut blurred = Mat::default();
        gaussian_blur(&gray, &mut blurred, blur_size, 0.0, 0.0, BORDER_DEFAULT)?;
        blurred
    } else {
        gray
    };
    let mut binary = Mat::default();
    threshold(&blurred_or_not, &mut binary, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU)?;
    let mut inv = Mat::default();
    bitwise_not(&binary, &mut inv, &no_array())?;
    Ok(inv)
}

fn deskew(img: &Mat) -> Result<Mat, Error> {
    let inv = binarize_inv(img, None)?;
    let mut coords = Vector::<Point>::default();
    find_non_zero(&inv, &mut coords)?;
    let rect = min_area_rect(&coords)?;
    let mut angle = rect.angle as f64;
    if angle < -45.0 {
        angle += 90.0;
    } else if angle > 45.0 {
        angle -= 90.0;
    }
    println!("angle : {:?}", angle);

    let center = Point2f::new((img.cols()/2) as f32, (img.rows()/2) as f32);
    let m = get_rotation_matrix_2d(center, angle, 1.0)?;
    let mut deskewed = Mat::default();
    warp_affine(&img, &mut deskewed, &m, img.size()?, INTER_CUBIC, BORDER_REPLICATE, Scalar::default())?;
    Ok(deskewed)
}

fn mask_lines_area(img: &Mat) -> Result<Mat, Error> {
    let binary_inv = binarize_inv(&img, Some(Size::new(9, 9)))?;
    let kernel = get_structuring_element(MORPH_RECT, Size::new(40, 2), Point::new(-1, -1))?;
    let mut dilated = Mat::default();
    dilate(&binary_inv, &mut dilated, &kernel, Point::new(-1, -1), 5, BORDER_CONSTANT, morphology_default_border_value()?)?;
    Ok(dilated)
}

fn separate_lines_area<F: FnMut(BoxedRef<Mat>) -> crate::Result<()>>(img: &Mat, mut handler: F) -> crate::Result<usize> {
    let mask = mask_lines_area(&img)?;
    let mut contours = Vector::<Mat>::default();
    find_contours(&mask, &mut contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point::new(0, 0))?;
    let lines = contours.len();
    for c in contours.iter().rev() {
        let rect = bounding_rect(&c)?;
        let roi = Mat::roi(img, rect)?;
        handler(roi)?;
    }
    Ok(lines)
}

fn text_to_lines_tensors<P: AsRef<Path>, F: FnMut(Tensor) -> crate::Result<()>>(p: P, config: &ImageTensorConfig, device: &Device, mut handler: F) -> crate::Result<usize> {
    let img = resize_if_needed(&p, config.max_width, config.max_height)?;
    let mut i = 0;
    let tmp_dir = tempdir()?;
    let filename = p.as_ref().file_name().unwrap_or(OsStr::new("line.jpg"));
    separate_lines_area(&img, |area| {
        let tmp_filename = format!("{}/{:03}-{}", &tmp_dir.path().display(), i, &filename.display());
        i += 1;
        imwrite(&tmp_filename, &area, &Vector::default())?;
        handler(load_image(&tmp_filename, config, device)?)
    })
}

pub(crate) fn text_to_lines_vec<P: AsRef<Path>>(p: P, config: &ImageTensorConfig, device: &Device) -> crate::Result<Vec<Tensor>> {
    let mut normalized_images = vec![];
    text_to_lines_tensors(p, config, device, |tensor| {
        normalized_images.push(tensor);
        Ok(())
    })?;
    Ok(normalized_images)
}

#[cfg(test)]
mod tests {

    use std::fs;

    use super::*;

    #[test]
    fn test_mask_lines_area() {
        let files = vec!["handwritten-text-en.jpg","20250926_171020.jpg", "20250930_221248.jpg", "20251001_094952.jpg", "printed-text-fr.png", "2023-03-26_22-18-04_UTC.jpg"];
        fs::create_dir_all("tests/generated-images/").unwrap();
        for file in files {
            let img = resize_if_needed(format!("dataset/{}", file), 1024, 768).unwrap();
            let area = mask_lines_area(&img).unwrap();
            let res = imwrite(&format!("tests/generated-images/mask-{}", file), &area, &Vector::default());
            assert!(res.is_ok() && res.unwrap());
        }
    }

    #[test]
    fn test_separate_lines_area() {
        let files = vec![("handwritten-text-en.jpg", 3), ("20250926_171020.jpg", 4), ("20251001_094952.jpg", 3), ("2023-03-26_22-18-04_UTC.jpg", 6)];
        fs::create_dir_all("tests/generated-images/").unwrap();
        for (file, nb_expected_area) in files {
            let img = resize_if_needed(format!("dataset/{}", file), 1024, 768).unwrap();
            let mut i = 0;
            let res = separate_lines_area(&img, |area| {
                let tmp_filename = format!("tests/generated-images/line-{:03}-{}", i, &file);
                i += 1;
                let res = imwrite(&tmp_filename, &area, &Vector::default());
                assert!(res.is_ok() && res.unwrap());
                Ok(())
            });
            assert!(res.is_ok());
            assert_eq!(res.unwrap(), nb_expected_area)
        }
    }

}
