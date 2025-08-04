# Usage
We include 4 models for watermark detection and 2 datasets for evaluation.
The `data` directory contains
1. `data/wm-nowm/valid` contain a balanced of around 3200 images for watermarked and non-watermarked images.
2. `data/datacomp-validation` contain 106 watermarked images and 849 non-watermarked images, annotated from CommonPool.
3. `data/laion_2b_subset` contains 176 watermarked images and 1132 non-watermarked images, annotated from LAION-2B.

For models,
1. `vlm` directory is used for evaluating `Rolm-OCR` and `Gemma-3-12b-it`
2. `mobilevit` directory is used for finetuning and evaluating a `MobileViTv2` model.
    * Trained on `data/wm-nowm/train`
3. For YoloV8, we follow the original [yolov8-scripts](https://github.com/MNeMoNiCuZ/yolov8-scripts) repository.
    * We modify the `generate.py` script to accomodate the `webdataset` format, included in `yolov8_modified` directory
    * To use the modified script, you need to set up `yolov8-scripts` following the original [yolov8-scripts](https://github.com/MNeMoNiCuZ/yolov8-scripts) repository.
