use std::marker::PhantomData;

use candle_transformers::{generation::LogitsProcessor, models::{mimi::{candle::{DType, Device, Tensor}, candle_nn::VarBuilder}, trocr::{self, TrOCRModel}, vit}};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

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
        self.model.reset_kv_cache();

        let mut logits_processor = LogitsProcessor::new(Self::SEED, None, None);
        let image = self.load_image(&image_item.path)?.unsqueeze(0)?.to_device(self.device)?;
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
        Ok(ImageItem {
            path: image_item.path,
            caption: image_item.caption,
            text_content: Some(text),
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
        assert!(&text.contains("boat"));
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
}
