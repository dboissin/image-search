use std::{error::Error, marker::PhantomData};

use candle_transformers::models::mimi::candle::Device;

use crate::{caption::CaptionStep, translation::TranslationStep};

pub mod caption;
pub mod translation;
pub mod ocr;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

trait State {}

struct Pending;
struct Captioned;

struct TextChecked;
struct Translated;
struct Completed;

impl State for Pending {}
impl State for Captioned {}
impl State for TextChecked {}
impl State for Translated {}
impl State for Completed {}

struct ImageItem<S: State> {
    path: String,
    caption: Option<String>,
    text_content: Option<String>,
    _state: PhantomData<S>
}

impl ImageItem<Pending> {

    pub fn new(image_path: &str) -> Self {
        Self { path: image_path.to_string(), caption: None, text_content: None, _state: PhantomData }
    }

}

pub struct ImageIndexer<'a> {
    caption_step: CaptionStep<'a>,
    translation_step: TranslationStep<'a>,
}

impl <'a> ImageIndexer<'a> {

    pub fn new(device: &'a Device) -> Result<Self> {
        Ok(Self {
            caption_step: CaptionStep::new(device)?,
            translation_step: TranslationStep::new(device)?
        })
    }

    pub fn indexing<S: AsRef<str>>(&mut self, image_paths: &[S]) -> Result<usize> {

        for image in image_paths {
            let img = ImageItem::new(image.as_ref());
            let img = self.caption_step.captioning(img)?;
            let img = self.translation_step.translate(img)?;
            if let Some(caption) = img.caption {
                println!("{} : {}", image.as_ref(), &caption);
            }
        }
        Ok(image_paths.len())
    }

}
