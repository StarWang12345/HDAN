# HDAN
## A Lightweight Heterogeneous Dual-Stream Disentanglement and Adaptive Refinement Network for single image super resolution
This is the official implementation code for A Lightweight Heterogeneous Dual-Stream Disentanglement and Adaptive Refinement Network for single image super resolution.
## Notes
2026-05: The first and preliminary version is realeased. Code may not be cleaned thoroughly, so feel free to open an issue if any question.
## Requirements
```
pip install -r requirements.txt
```
## Training
1. Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](https://github.com/LimBee/NTIRE2017) from [Google Drive](https://drive.google.com/drive/folders/1B-uaxvV9qeuQ-t7MFiN1oEdA6dKnj2vW?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1CFIML6KfQVYGZSNFrhMXmA)
2. To build a DF2K dataset in H5 format, place the images from DIV2K_train_HR and Flickr2K_HR in the same folder DF2K_train_HR, then modify the path in DIVFIRh5.py according to your specific setup and run the script.The resulting file will be named something like DF2K_x2_fixed.h5.
```
python DIVFIRh5.py
```

3. Run Training.
```
python train.py
```
## Testing
1. Download the five test datasets (Set5, Set14, B100, Urban100, Manga109) from [Google Drive](https://drive.google.com/drive/folders/1lsoyAjsUEyp7gm1t6vZI9j7jr9YzKzcF?usp=sharing) 
2. Pretrained models have be placed in ./pretrained_models/ folder.
3. Run test.
```
python sample.py
```
## Citation
If you find this repository/work helpful in your research, welcome to cite the paper and give a star.
