
use std::{marker::PhantomData};

use candle_transformers::{generation::LogitsProcessor, models::{blip::Config, mimi::candle::{self, Device, Tensor}, quantized_blip::BlipForConditionalGeneration}, quantized_var_builder::VarBuilder};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

use crate::{Captioned, ImageItem, Pending, Result};

pub(crate) struct CaptionStep<'a> {
    model: BlipForConditionalGeneration,
    tokenizer: Tokenizer,
    device: &'a Device,
}

impl <'a> CaptionStep<'a> {

    const SEP_TOKEN_ID: u32 = 102;
    const MAX_TOKENS_ITER: usize = 1000;
    const SEED: u64 = 42;
    const SHAPE: (usize, usize, usize) = (384, 384, 3);
    const CHANNEL_NORMALIZATION_MEAN: &'static[f32] = &[0.48145466f32, 0.4578275, 0.40821073];
    const CHANNEL_NORMALIZATION_STD_DEV: &'static[f32] = &[0.26862954f32, 0.261_302_6, 0.275_777_1];

    pub(crate) fn new(device: &'a Device) -> Result<Self> {
        let api = Api::new()?;
        let blip_id = "Salesforce/blip-image-captioning-large".to_string();
        let tokenizer_file = api.model(blip_id).get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_file)?;
        let config = Config::image_captioning_large();

        let model_id = "lmz/candle-blip".to_string();
        let model_file = api.model(model_id).get("blip-image-captioning-large-q4k.gguf")?;

        let vb = VarBuilder::from_gguf(model_file, device)?;
        let model = BlipForConditionalGeneration::new(&config, vb)?;

        Ok(Self { model, tokenizer, device })
    }

    fn load_image<P: AsRef<std::path::Path>>(&self, p: P) -> Result<Tensor> {
        let img = image::ImageReader::open(p)?
            .decode()?
            .resize_to_fill(Self::SHAPE.0 as u32, Self::SHAPE.1 as u32, image::imageops::FilterType::Triangle);
        let img = img.to_rgb8();
        let data = img.into_raw();
        let data = Tensor::from_vec(data, Self::SHAPE, self.device)?.permute((2, 0, 1))?;
        let mean = Tensor::new(Self::CHANNEL_NORMALIZATION_MEAN, self.device)?.reshape((3, 1, 1))?;
        let std = Tensor::new(Self::CHANNEL_NORMALIZATION_STD_DEV, self.device)?.reshape((3, 1, 1))?;

        Ok((data.to_dtype(candle::DType::F32)? / 255.)?.broadcast_sub(&mean)?.broadcast_div(&std)?)
    }

    pub(crate) fn captioning(&mut self, image_item: crate::ImageItem<Pending>) -> Result<crate::ImageItem<Captioned>> {
        self.model.reset_kv_cache();
        let image = self.load_image(&image_item.path)?.to_device(self.device)?;
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
        assert!(res.is_ok());
        let caption = res.unwrap().caption.unwrap();
        assert!(&caption.contains("boat"));
    }

}
