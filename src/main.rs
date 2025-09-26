use std::env;

use candle_transformers::models::mimi::candle::Device;
use image_search::ImageIndexer;

fn main() {
    let args: Vec<String> = env::args().collect();

    let device = Device::cuda_if_available(0).unwrap();
    let mut indexer = ImageIndexer::new(&device).unwrap();
    let _ = indexer.indexing(&args[1..]).unwrap();
}
