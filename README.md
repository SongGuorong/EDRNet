# EDRNet

source code for our IEEE TIM 2020 paper entitled [EDRNet: Encoder-Decoder Residual Network for Salient Object Detection of Strip Steel Surface Defects](https://ieeexplore.ieee.org/document/9116810) (**DOI：10.1109/TIM.2020.3002277**) by Guorong Song, Kechen Song and Yunhui Yan.

## Requirement

- Python 3.6
- Pytorch 0.4.1 or 1.0.1(**default**)
- numpy
- torchvision
- glob
- PIL
- scikit-image

**This code is tested on Ubuntu 16.04.** 

## Training

1. cd to `./Data`, and Unzip the file of `trainingDataset.zip` into this folder.
2. **path of training images:**`./Data/trainingDataset/imgs_train/` **path of training labels:**`./Data/trainingDataset/masks_train/`
3. run`python edrnet_train.py` to start training
4. the trained model will be saved in `./trained_models`

## Testing

1. download the test dataset [SD-saliency-900.zip](https://drive.google.com/file/d/1yQdfow1-WvDilQTZ1zj1EbbErN1DksVF/view?usp=sharing), then Unzip it to the directory of `./Data`
2. download the pre-trained model [EDRNet_epoch_600.pth](https://drive.google.com/file/d/1FJe9j-F7r3kdlEJgBC-Izi37ANytwLF-/view?usp=sharing), then put it to the directory of `./trained_models`
3. **path of testing dataset:** `./Data/SD-saliency-900/imgs/` **path of pre-trained model:** `./trained_models/EDRNet_epoch_600.pth`
4. run`python edrnet_test.py` to start testing  
5. the predict results will be saved in `./Data/test_results/`


**Note: If you use `SD-saliency-900` dataset in your paper, please cite [Saliency detection for strip steel surface defects using multiple constraints and improved texture features](https://www.sciencedirect.com/science/article/abs/pii/S0143816619317361)**

## Results

We also provide the experimental results of all the comparative methods in our paper.([Results](https://drive.google.com/file/d/1XAFLIPbgJQpX2QiL2JZtnoK0QY2ARWTn/view?usp=sharing))

**You can also download all the files including `SD-saliency-900.zip, EDRNet_epoch_600.pth, Results` in BaiduYun Drive.(link:https://pan.baidu.com/s/1RSgkzNKxXA11ajtoFnk6Mw     code: z91m)**

## Performance Preview

**Visual comparison**
![visual_comparison.jpg](https://storage.live.com/items/72AB557781850244!8964?authkey=AFAbVUdk1jyWqZA)

**Quantitative comparison**
![quantitative_evaluation.png](https://storage.live.com/items/72AB557781850244!8966?authkey=AFAbVUdk1jyWqZA)

## Citation

```
@InProceedings{SGR_2020_TIM,
author = {Song, Guorong and Song, Kechen and Yan, Yunhui},
title = {EDRNet: Encoder-Decoder Residual Network for Salient Object Detection of Strip Steel Surface Defects},
booktitle = {IEEE Transactions on Instrumentation & Measurement (IEEE TIM)},
month = {June},
year = {2020}
}
```
