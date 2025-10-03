use std::{collections::HashMap, marker::PhantomData, path::Path};

use candle_transformers::{generation::LogitsProcessor, models::{mimi::{candle::{DType, Device, Tensor}, candle_nn::VarBuilder}, trocr::{self, TrOCRModel}, vit}};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

use crate::{preprocessing::{self, ImageTensorConfig, ResizeStrategy}, Captioned, ImageItem, Result, TextChecked, TextLanguage};

#[derive(Debug, Clone, serde::Deserialize)]
struct Config {
    encoder: vit::Config,
    decoder: trocr::TrOCRConfig,
}

struct ModelItem {
    config: Config,
    model: TrOCRModel,
    decoder: Tokenizer,
    img_config: ImageTensorConfig,
}

pub(crate) struct TextExtractionStep<'a> {
    models: HashMap<TextLanguage, ModelItem>,
    device: &'a Device,
}

impl <'a> TextExtractionStep<'a> {

    const MAX_TOKENS_ITER: usize = 1000;
    const SEED: u64 = 42;
    const MODEL_ID: (&'static str, &'static str) = ("microsoft/trocr-base-handwritten","");
    // const FR_MODEL_ID: (&'static str, &'static str) = ("agomberto/trocr-large-handwritten-fr","");
    const TOKENIZER_ID: (&'static str, &'static str) = ("ToluClassics/candle-trocr-tokenizer", "tokenizer.json");

    pub(crate) fn new(device: &'a Device) -> Result<Self> {
        let mut models = HashMap::new();
        let api = Api::new()?;

        let config_file = api.model(Self::MODEL_ID.0.to_string()).get("config.json")?;
        let config: Config = serde_json::from_reader(std::fs::File::open(config_file)?)?;

        let mut img_config = ImageTensorConfig::default();
        img_config.resize_strategy = ResizeStrategy::Exact;

        let tokenizer_decode_file = api.model(Self::TOKENIZER_ID.0.to_string()).get(Self::TOKENIZER_ID.1)?;
        let decoder = Tokenizer::from_file(tokenizer_decode_file)?;

        let model_file = api.model(Self::MODEL_ID.0.to_string()).get("model.safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&model_file], DType::F32, &device)? };
        let model = TrOCRModel::new(&config.encoder, &config.decoder, vb)?;

        models.insert(TextLanguage::EN, ModelItem { config, model, decoder, img_config });

        let base_path = Path::new("/home/dboissin/ia/models/trocr-fr");
        let config_file = base_path.join("config.json");
        let config: Config = serde_json::from_reader(std::fs::File::open(config_file)?)?;

        let mut img_config = ImageTensorConfig::default();
        img_config.resize_strategy = ResizeStrategy::Exact;

        let tokenizer_decode_file = base_path.join("tokenizer.json");
        let decoder = Tokenizer::from_file(tokenizer_decode_file)?;

        let model_file = base_path.join("model.safetensors");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&model_file], DType::F32, &device)? };
        let model = TrOCRModel::new(&config.encoder, &config.decoder, vb)?;

        models.insert(TextLanguage::FR, ModelItem { config, model, decoder, img_config });

        Ok(Self { models, device })
    }

    pub(crate) fn read_text(&mut self, image_item: crate::ImageItem<Captioned>) -> Result<crate::ImageItem<TextChecked>> {
        let model_option = image_item.text_language.as_ref().map(|language| self.models.get_mut(language)).flatten();
        if model_option.is_none() {
            println!("No language defined or language not supported. Skip OCR.");
            return Ok(ImageItem {
                path: image_item.path,
                caption: image_item.caption,
                text_content: image_item.text_content,
                text_language: image_item.text_language,
                _state: PhantomData
            })
        }
        let model_item = model_option.unwrap();
        let model = &mut model_item.model;
        let config = &model_item.config;
        let decoder = &model_item.decoder;

        let normalized_images = preprocessing::text_to_lines_vec(&image_item.path, &model_item.img_config, &self.device)?;
        let mut result_text = vec![];
        for image in &normalized_images {
            model.reset_kv_cache();

            let mut logits_processor = LogitsProcessor::new(Self::SEED, None, None);
            let image = image.unsqueeze(0)?.to_device(self.device)?;
            let encoder_xs = model.encoder().forward(&image)?;

            let mut token_ids: Vec<u32> = vec![config.decoder.decoder_start_token_id];
            for index in 0..Self::MAX_TOKENS_ITER {
                let context_size = if index >= 1 { 1 } else { token_ids.len() };
                let start_pos = token_ids.len().saturating_sub(context_size);
                let input_ids = Tensor::new(&token_ids[start_pos..], &self.device)?.unsqueeze(0)?;

                let logits = model.decode(&input_ids, &encoder_xs, start_pos)?;

                let logits = logits.squeeze(0)?;
                let logits = logits.get(logits.dim(0)? - 1)?;
                let token = logits_processor.sample(&logits)?;
                token_ids.push(token);

                if token == config.decoder.eos_token_id {
                    break;
                }
            }

            let text = decoder.decode(&token_ids, true)?;
            result_text.push(text);
        }

        Ok(ImageItem {
            path: image_item.path,
            caption: image_item.caption,
            text_content: Some(result_text.join("\n")),
            text_language: image_item.text_language,
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
            text_language: Some(TextLanguage::EN),
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
            text_language: Some(TextLanguage::EN),
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
            text_language: Some(TextLanguage::EN),
            _state: PhantomData
        };
        let device = Device::cuda_if_available(0).unwrap();
        let mut ocr = TextExtractionStep::new(&device).unwrap();
        let res = ocr.read_text(img_item).unwrap();
        let text = res.text_content.unwrap();
        println!("{text}");
        assert!(&text.to_lowercase().contains("to separate lines before"));
    }

    #[test]
    fn test_multi_line_handwritten_fr() {
        let img_item = ImageItem {
            path: "dataset/printed-text-fr.png".to_string(),
            caption: None,
            text_content: None,
            text_language: Some(TextLanguage::FR),
            _state: PhantomData
        };
        let device = Device::cuda_if_available(0).unwrap();
        let mut ocr = TextExtractionStep::new(&device).unwrap();
        let res = ocr.read_text(img_item).unwrap();
        let text = res.text_content.unwrap();
        println!("{text}");
        assert!(&text.contains("habituellement une bonne valeur"));
    }

    #[test]
    fn test_multi_lines_handwritten_fr_multi_colors() {
        let img_item = ImageItem {
            path: "dataset/20251001_094952.jpg".to_string(),
            caption: None,
            text_content: None,
            text_language: Some(TextLanguage::FR),
            _state: PhantomData
        };
        let device = Device::cuda_if_available(0).unwrap();
        let mut ocr = TextExtractionStep::new(&device).unwrap();
        let res = ocr.read_text(img_item).unwrap();
        let text = res.text_content.unwrap();
        println!("{text}");
        assert!(&text.contains("en français"));
    }

}
