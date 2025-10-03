
use std::{marker::PhantomData};

use candle_transformers::{generation::LogitsProcessor, models::{blip::Config, mimi::{candle::{DType, Device, Tensor}, candle_nn::VarBuilder}, blip::BlipForConditionalGeneration}};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

use crate::{preprocessing::{self, ImageTensorConfig, ResizeStrategy}, Captioned, ImageItem, Pending, Result};

pub(crate) struct CaptionStep<'a> {
    model: BlipForConditionalGeneration,
    tokenizer: Tokenizer,
    device: &'a Device,
    img_config: ImageTensorConfig,
}

impl <'a> CaptionStep<'a> {

    const SEP_TOKEN_ID: u32 = 102;
    const MAX_TOKENS_ITER: usize = 1000;
    const SEED: u64 = 42;
    const MODEL_ID: &'static str = "Salesforce/blip-image-captioning-large";

    pub(crate) fn new(device: &'a Device) -> Result<Self> {
        let api = Api::new()?;
        let tokenizer_file = api.model(Self::MODEL_ID.to_string()).get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_file)?;
        let config = Config::image_captioning_large();

        let mut img_config = ImageTensorConfig::default();
        img_config.resize_strategy = ResizeStrategy::Pad;
        img_config.normalization_mean =  vec![0.48145466, 0.4578275, 0.40821073];
        img_config.normalization_std_dev = vec![0.26862954, 0.26130258, 0.27577711];
        img_config.resample = 3;

        let model_file = api.model(Self::MODEL_ID.to_string()).get("model.safetensors")?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&model_file], DType::F32, &device)? };
        let model = BlipForConditionalGeneration::new(&config, vb)?;

        Ok(Self { model, tokenizer, device, img_config })
    }

    pub(crate) fn captioning(&mut self, image_item: crate::ImageItem<Pending>) -> Result<crate::ImageItem<Captioned>> {
        self.model.reset_kv_cache();
        let image = preprocessing::load_image(&image_item.path, &self.img_config, self.device)?.to_device(self.device)?;
        let image_embeds = image.unsqueeze(0)?.apply(self.model.vision_model())?;
        let mut logits_processor = LogitsProcessor::new(CaptionStep::SEED, None, None);

        let mut token_ids = vec![30522u32];
        for index in 0..CaptionStep::MAX_TOKENS_ITER {
            let context_size = if index > 0 { 1 } else { token_ids.len() };
            let start_pos = token_ids.len().saturating_sub(context_size);
            let input_ids = Tensor::new(&token_ids[start_pos..], self.device)?.unsqueeze(0)?;
            let logits = self.model.text_decoder().forward(&input_ids, &image_embeds)?;
            let logits = logits.squeeze(0)?;
            let logits = logits.get(logits.dim(0)? - 1)?;
            let token = logits_processor.sample(&logits)?;
            if token == CaptionStep::SEP_TOKEN_ID {
                break;
            }
            token_ids.push(token);
        }

        let caption = self.tokenizer.decode(&token_ids, true)?;
        println!("caption en anglais : {}", &caption);
        Ok(ImageItem {
            path: image_item.path,
            caption: Some(caption),
            text_content: image_item.text_content,
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
    fn test_image_captionning() {
        let img_item = ImageItem::new("dataset/20250816_123851.jpg");
        let device = Device::cuda_if_available(0).unwrap();
        let mut caption = CaptionStep::new(&device).unwrap();
        let res = caption.captioning(img_item);
        let caption = res.unwrap().caption.unwrap();
        assert!(&caption.contains("boat"));
    }

}
