use std::marker::PhantomData;

use candle_transformers::{generation::LogitsProcessor, models::{marian::{Config, MTModel}, mimi::{candle::{DType, Device, Tensor}, candle_nn::VarBuilder}}};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

use crate::{Captioned, ImageItem, Translated, Result};


pub(crate) struct TranslationStep<'a> {
    config: Config,
    model: MTModel,
    encoder: Tokenizer,
    decoder: Tokenizer,
    device: &'a Device,
}

impl <'a> TranslationStep<'a> {

    const MAX_TOKENS_ITER: usize = 1000;
    const SEED: u64 = 42;
    const MODEL_ID: (&'static str, &'static str) = ("Helsinki-NLP/opus-mt-en-fr", "refs/pr/9");
    const TOKENIZER_ID: (&'static str, &'static str, &'static str) = ("lmz/candle-marian", "tokenizer-marian-base-en.json", "tokenizer-marian-base-fr.json");

    pub(crate) fn new(device: &'a Device) -> Result<Self> {
        let api = Api::new()?;
        let config = Config::opus_mt_en_fr();
        let model_file = api.repo(hf_hub::Repo::with_revision(
            Self::MODEL_ID.0.to_string(), hf_hub::RepoType::Model,Self::MODEL_ID.1.to_string()))
            .get("model.safetensors")?;
        let tokenizer_encode_file = api.model(Self::TOKENIZER_ID.0.to_string()).get(Self::TOKENIZER_ID.1)?;
        let tokenizer_decode_file = api.model(Self::TOKENIZER_ID.0.to_string()).get(Self::TOKENIZER_ID.2)?;
        let encoder = Tokenizer::from_file(tokenizer_encode_file)?;
        let decoder = Tokenizer::from_file(tokenizer_decode_file)?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&model_file], DType::F32, &device)? };
        let model = MTModel::new(&config, vb)?;

        Ok(Self { config, model, encoder, decoder, device })
    }

    pub(crate) fn translate(&mut self, image_item: crate::ImageItem<Captioned>) -> Result<crate::ImageItem<Translated>> {
        self.model.reset_kv_cache();

        let mut logits_processor = LogitsProcessor::new(Self::SEED, None, None);

        let encoder_xs = {
            let mut tokens = self.encoder.encode(image_item.caption.unwrap(), true)?.get_ids().to_vec();
            tokens.push(self.config.eos_token_id);
            let tokens = Tensor::new(tokens.as_slice(), self.device)?.unsqueeze(0)?;
            self.model.encoder().forward(&tokens, 0)?
        };

        let mut token_ids = vec![self.config.decoder_start_token_id];
        for index in 0..Self::MAX_TOKENS_ITER {
            let context_size = if index >= 1 { 1 } else { token_ids.len() };
            let start_pos = token_ids.len().saturating_sub(context_size);
            let input_ids = Tensor::new(&token_ids[start_pos..], self.device)?.unsqueeze(0)?;
            let logits = self.model.decode(&input_ids, &encoder_xs, start_pos)?;
            let logits = logits.squeeze(0)?;
            let logits = logits.get(logits.dim(0)? - 1)?;
            let token = logits_processor.sample(&logits)?;
            token_ids.push(token);
            if token == self.config.eos_token_id || token == self.config.forced_eos_token_id {
                break;
            }
        }
        let caption_fr = self.decoder.decode(&token_ids, true)?
            .trim_start_matches("<NIL>").trim_end_matches("</s>").to_string();
        Ok(ImageItem {
            path: image_item.path,
            caption: Some(caption_fr),
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
    fn test_translate_en_fr() {
        let text = ImageItem {
            path: "".to_string(),
            caption: Some("there are many boats docked at the dock on the water".to_string()),
            text_content: None,
            _state: PhantomData
        };
        let device = Device::cuda_if_available(0).unwrap();
        let mut translation = TranslationStep::new(&device).unwrap();
        let res = translation.translate(text);
        assert!(res.is_ok());
        let caption = res.unwrap().caption.unwrap();
        println!("{}", &caption);
        assert!(&caption.contains("boat"));
    }

}
