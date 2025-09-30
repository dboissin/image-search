use std::marker::PhantomData;

use candle_transformers::{generation::LogitsProcessor, models::{mimi::{candle::{DType, Device, Tensor}, candle_nn::VarBuilder}, trocr::{self, TrOCRModel}, vit}};
use hf_hub::api::sync::Api;
use tempfile::tempdir;
use tokenizers::Tokenizer;
use opencv::{core::*, imgcodecs, imgproc};

use crate::{Captioned, ImageItem, Result, TextChecked};

#[derive(Debug, Clone, serde::Deserialize)]
struct Config {
    encoder: vit::Config,
    decoder: trocr::TrOCRConfig,
}

pub(crate) struct TextExtractionStep<'a> {
    config: Config,
    model: TrOCRModel,
    decoder: Tokenizer,
    device: &'a Device,
}

impl <'a> TextExtractionStep<'a> {

    const MAX_TOKENS_ITER: usize = 1000;
    const SEED: u64 = 42;
    const MODEL_ID: (&'static str, &'static str) = ("microsoft/trocr-base-handwritten",""); // agomberto/trocr-base-printed-fr",""); //("agomberto/trocr-large-handwritten-fr", "refs/pr/3");
    const TOKENIZER_ID: (&'static str, &'static str) = ("ToluClassics/candle-trocr-tokenizer", "tokenizer.json");
    const SHAPE: (usize, usize, usize) = (384, 384, 3);
    const CHANNEL_NORMALIZATION_MEAN: &'static[f32] = &[0.5, 0.5, 0.5];
    const CHANNEL_NORMALIZATION_STD_DEV: &'static[f32] = &[0.5, 0.5, 0.5];

    fn load_image<P: AsRef<std::path::Path>>(&self, p: P) -> Result<Tensor> {
        let img = image::ImageReader::open(p)?
            .decode()?
            .resize_exact(Self::SHAPE.0 as u32, Self::SHAPE.1 as u32, image::imageops::FilterType::Triangle);
        let img = img.to_rgb8();
        let data = img.into_raw();
        let data = Tensor::from_vec(data, Self::SHAPE, self.device)?.permute((2, 0, 1))?;
        let mean = Tensor::new(Self::CHANNEL_NORMALIZATION_MEAN, self.device)?.reshape((3, 1, 1))?;
        let std = Tensor::new(Self::CHANNEL_NORMALIZATION_STD_DEV, self.device)?.reshape((3, 1, 1))?;

        Ok((data.to_dtype(DType::F32)? / 255.)?.broadcast_sub(&mean)?.broadcast_div(&std)?)
    }

    fn normalize(&self, data: Vec<u8>) -> Result<Tensor> {
        let data = Tensor::from_vec(data, Self::SHAPE, self.device)?.permute((2, 0, 1))?;
        let mean = Tensor::new(Self::CHANNEL_NORMALIZATION_MEAN, self.device)?.reshape((3, 1, 1))?;
        let std = Tensor::new(Self::CHANNEL_NORMALIZATION_STD_DEV, self.device)?.reshape((3, 1, 1))?;

        Ok((data.to_dtype(DType::F32)? / 255.)?.broadcast_sub(&mean)?.broadcast_div(&std)?)
    }

    fn preprocess<P: AsRef<str>>(&self, p: P) -> Result<Vec<Tensor>> {
        let filename = std::path::Path::new(p.as_ref()).file_name().unwrap();
        let img = opencv::imgcodecs::imread(p.as_ref(), imgcodecs::IMREAD_COLOR)?;

        let mut gray = Mat::default();
        imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

        // imgcodecs::imwrite(&format!("{}-gray.jpg", p.as_ref()), &gray, &Vector::default())?;

        let mut binary = Mat::default();
        imgproc::threshold(&gray, &mut binary, 0.0, 255.0, imgproc::THRESH_BINARY | imgproc::THRESH_OTSU)?;

        // imgcodecs::imwrite(&format!("{}-binary.jpg", p.as_ref()), &binary, &Vector::default())?;

        let mut inv = Mat::default();
        opencv::core::bitwise_not(&binary, &mut inv, &no_array())?;
        // imgcodecs::imwrite(&format!("{}-inv.jpg", p.as_ref()), &inv, &Vector::default())?;

        let mut coords = Vector::<Point>::default();
        opencv::core::find_non_zero(&inv, &mut coords)?;
        let rect = imgproc::min_area_rect(&coords)?;
        let mut angle = rect.angle as f64;
        if angle < -45.0 {
            angle += 90.0;
        } else if angle > 45.0 {
            angle -= 90.0;
        }

        println!("angle : {:?}", angle);

        let center = Point2f::new((img.cols()/2) as f32, (img.rows()/2) as f32);
        let m = imgproc::get_rotation_matrix_2d(center, angle, 1.0)?;
        let mut deskewed = Mat::default();
        imgproc::warp_affine(
            &img, &mut deskewed, &m, img.size()?,
            imgproc::INTER_CUBIC,
            opencv::core::BORDER_REPLICATE,
            Scalar::default()
        )?;

        // imgcodecs::imwrite(&format!("{}-deskewed.jpg", p.as_ref()), &deskewed, &Vector::default())?;

        let mut gray_deskewed = Mat::default();
        imgproc::cvt_color(&img, &mut gray_deskewed, imgproc::COLOR_BGR2GRAY, 0)?;

        let mut blurred = Mat::default();
        imgproc::gaussian_blur(&gray_deskewed, &mut blurred, Size::new(7, 7), 0.0, 0.0, opencv::core::BORDER_DEFAULT)?;

        let mut binary_deskewed = Mat::default();
        imgproc::threshold(&blurred, &mut binary_deskewed, 0.0, 255.0, imgproc::THRESH_BINARY_INV | imgproc::THRESH_OTSU)?;

        // imgcodecs::imwrite(&format!("{}-binary2.jpg", p.as_ref()), &binary2, &Vector::default())?;

        let kernel = imgproc::get_structuring_element(
            imgproc::MORPH_RECT, Size::new(50, 1), Point::new(-1, -1))?;
        let mut dilated = Mat::default();
        imgproc::dilate(&binary_deskewed, &mut dilated, &kernel, Point::new(-1, -1), 5,
            opencv::core::BORDER_CONSTANT, imgproc::morphology_default_border_value()?)?;

        // imgcodecs::imwrite(&format!("{}-kernel.jpg", p.as_ref()), &dilated, &Vector::default())?;

        let mut contours = Vector::<Mat>::default();
        imgproc::find_contours(
            &dilated, &mut contours, imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0))?;

        let tmp_dir = tempdir()?;
        let mut normalized_images = vec![];
        for (i, c) in contours.iter().rev().enumerate() {
            let rect = imgproc::bounding_rect(&c)?;
            let roi = Mat::roi(&deskewed, rect)?;

            // let data: Vec<u8> = roi.to_vec_2d::<Vec3b>().into_iter().flatten().flatten().flat_map(|pix| pix.0).collect();
            // normalized_images.push(self.normalize(data)?);
            let filename = format!("{}/{:03}-{}", &tmp_dir.path().display(), i, &filename.display());
            println!("{filename}");
            // TODO check rotation on single line
            imgcodecs::imwrite(&filename.as_ref(), &roi, &Vector::default())?;
            normalized_images.push(self.load_image(filename)?);
        }
        //Ok(Tensor::stack(&normalized_images, 0)?)
        Ok(normalized_images)
    }

    pub(crate) fn new(device: &'a Device) -> Result<Self> {
        let api = Api::new()?;
        let config_file = api.model(Self::MODEL_ID.0.to_string()).get("config.json")?;
        // let config_file = api.repo(hf_hub::Repo::with_revision(
        //     Self::MODEL_ID.0.to_string(), hf_hub::RepoType::Model,Self::MODEL_ID.1.to_string()))
        //     .get("config.json")?;
        let config: Config = serde_json::from_reader(std::fs::File::open(config_file)?)?;
        let model_file = api.model(Self::MODEL_ID.0.to_string()).get("model.safetensors")?;
        // let model_file = api.repo(hf_hub::Repo::with_revision(
        //     Self::MODEL_ID.0.to_string(), hf_hub::RepoType::Model,Self::MODEL_ID.1.to_string()))
        //     .get("model.safetensors")?;
        // let tokenizer_decode_file = api.model(Self::MODEL_ID.0.to_string()).get("tokenizer.json")?;
        let tokenizer_decode_file = api.model(Self::TOKENIZER_ID.0.to_string()).get(Self::TOKENIZER_ID.1)?;
        let decoder = Tokenizer::from_file(tokenizer_decode_file)?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&model_file], DType::F32, &device)? };
        let model = TrOCRModel::new(&config.encoder, &config.decoder, vb)?;

        Ok(Self { config, model, decoder, device })
    }

    pub(crate) fn read_text(&mut self, image_item: crate::ImageItem<Captioned>) -> Result<crate::ImageItem<TextChecked>> {
        let normalized_images = self.preprocess(&image_item.path)?;
        let mut result_text = vec![];
        for image in &normalized_images {
            self.model.reset_kv_cache();

            let mut logits_processor = LogitsProcessor::new(Self::SEED, None, None);
            // let image = self.load_image(&image_item.path)?.unsqueeze(0)?.to_device(self.device)?;
            let image = image.unsqueeze(0)?.to_device(self.device)?;
            let encoder_xs = self.model.encoder().forward(&image)?;

            let mut token_ids: Vec<u32> = vec![self.config.decoder.decoder_start_token_id];
            for index in 0..Self::MAX_TOKENS_ITER {
                let context_size = if index >= 1 { 1 } else { token_ids.len() };
                let start_pos = token_ids.len().saturating_sub(context_size);
                let input_ids = Tensor::new(&token_ids[start_pos..], &self.device)?.unsqueeze(0)?;

                let logits = self.model.decode(&input_ids, &encoder_xs, start_pos)?;

                let logits = logits.squeeze(0)?;
                let logits = logits.get(logits.dim(0)? - 1)?;
                let token = logits_processor.sample(&logits)?;
                token_ids.push(token);

                if token == self.config.decoder.eos_token_id {
                    break;
                }
            }

            let text = self.decoder.decode(&token_ids, true)?;
            println!("text : {}", &text);
            result_text.push(text);
        }

        Ok(ImageItem {
            path: image_item.path,
            caption: image_item.caption,
            text_content: Some(result_text.join("\n")),
            _state: PhantomData
        })
    }

}


#[cfg(test)]
mod tests {

    use candle_transformers::models::mimi::candle::Device;

    use super::*;

    #[test]
    fn test_single_line_handwritten_en() {
        let img_item = ImageItem {
            path: "dataset/line-handwritten-text-en.jpg".to_string(),
            caption: None,
            text_content: None,
            _state: PhantomData
        };
        let device = Device::cuda_if_available(0).unwrap();
        let mut ocr = TextExtractionStep::new(&device).unwrap();
        let res = ocr.read_text(img_item);
        assert!(res.is_ok());
        let text = res.unwrap().text_content.unwrap();
        assert!(&text.contains("This is a handwritten"));
    }

    #[test]
    fn test_multi_line_handwritten_en() {
        let img_item = ImageItem {
            path: "dataset/handwritten-text-en.jpg".to_string(),
            caption: None,
            text_content: None,
            _state: PhantomData
        };
        let device = Device::cuda_if_available(0).unwrap();
        let mut ocr = TextExtractionStep::new(&device).unwrap();
        let res = ocr.read_text(img_item).unwrap();
        let text = res.text_content.unwrap();
        println!("{text}");
        assert!(&text.contains("as good as you can"));
    }

    #[test]
    fn test_multi_lines_handwritten_en_multi_colors() {
        let img_item = ImageItem {
            path: "dataset/20250930_221248.jpg".to_string(),
            caption: None,
            text_content: None,
            _state: PhantomData
        };
        let device = Device::cuda_if_available(0).unwrap();
        let mut ocr = TextExtractionStep::new(&device).unwrap();
        let res = ocr.read_text(img_item).unwrap();
        let text = res.text_content.unwrap();
        println!("{text}");
        assert!(&text.contains("to separate lines before"));
    }

    #[test]
    fn test_single_line_handwritten_fr() {
        let img_item = ImageItem {
            path: "dataset/20250926_171020.jpg".to_string(),
            caption: None,
            text_content: None,
            _state: PhantomData
        };
        let device = Device::cuda_if_available(0).unwrap();
        let mut ocr = TextExtractionStep::new(&device).unwrap();
        let res = ocr.read_text(img_item).unwrap();
        let text = res.text_content.unwrap();
        assert!(&text.contains("boat"));
    }

    #[test]
    fn test_preprocess_multi_line_handwritten_en() {
        let img_item:ImageItem<Captioned> = ImageItem {
            path: "dataset/handwritten-text-en.jpg".to_string(),
            caption: None,
            text_content: None,
            _state: PhantomData
        };
        let device = Device::cuda_if_available(0).unwrap();
        let ocr = TextExtractionStep::new(&device).unwrap();
        let res = ocr.preprocess(&img_item.path);
        assert!(res.is_ok());
    }

    #[test]
    fn test_preprocess_single_line_handwritten_fr() {
        let img_item:ImageItem<Captioned> = ImageItem {
            path: "dataset/20250926_171020.jpg".to_string(),
            caption: None,
            text_content: None,
            _state: PhantomData
        };
        let device = Device::cuda_if_available(0).unwrap();
        let ocr = TextExtractionStep::new(&device).unwrap();
        let res = ocr.preprocess(&img_item.path);
        assert!(res.is_ok());
    }

}
