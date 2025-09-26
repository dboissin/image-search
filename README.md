Image-Search
============

Project to test candle inference library with images.

Build and Run
-------------

```bash
 cargo build --release
 target/release/image-search dataset/20250404_084255.jpg dataset/20250816_123851.jpg
```

Output :
```bash
caption en anglais : arafed view of a park with benches and a church in the background
dataset/20250404_084255.jpg :  vue araftée d'un parc avec bancs et une église en arrière-plan
caption en anglais : boats are docked at a marina with a few people walking by
dataset/20250816_123851.jpg :  bateaux sont amarrés à une marina avec quelques personnes à pied
```
