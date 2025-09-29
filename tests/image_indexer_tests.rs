use candle_transformers::models::mimi::candle::Device;
use image_search::ImageIndexer;


#[test]
fn test_image_captionning() {
    let img = "dataset/20250816_123851.jpg";
    let device = Device::cuda_if_available(0).unwrap();
    let mut indexer = ImageIndexer::new(&device).unwrap();
    let res = indexer.indexing(&[img]);
    assert!(res.is_ok());
}
