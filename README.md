# MMAA: Multi-Modal Attribute-Aware framework for intelligent medical report generation
This repository provides code for the paper, Automatic Medical Report Production Based on Collaborative Learning of Medical Image Features and Clinical Symptom Attributes. Please refer to our paper for better understanding the method.
## Pipeline
![image](https://github.com/stellating/MMAA/blob/main/img/1653319285.jpg)
## Getting started
### Environments
* python 3.5
* pytorch 1.8.1
* torchvision 0.9.1
* CUDA 11.1
### Packages
* numpy
* pandas
* time
* pillow
* json
### Datasets
Download from [IU X-Ray](https://drive.google.com/drive/folders/186KDV48o-jtK09b4yzNHbfo-kQDUyEsw?usp=sharing) or [baidupandisk](https://pan.baidu.com/s/1fdpo12x0YgcXJ62a1K3-Gg) with the password of '62ui', prepare dataset in data directory as follows.
```
MMAA
│   lib
|   models
│   modules
└───data
│   │   iu_xray
│   │   │   images
            └───images_normalized
│   │   │   iuxray_label_40_annotation.json
└───README.md
```
### Running
__0.Clone this repo:__  
```
  git clone https://github.com/stellating/MMAA.git
  cd MMAA
```
__1.Train:__  
Train a model on the IU X-Ray data.
```
sh run_iu_xray.sh 
``` 
__2.Test:__  
The pretrained model can be download from [here]() or [BaiduNetdisk](https://pan.baidu.com/s/1BWE3V2WPjB8ffu9j8ri2_Q) with the password of 'duz8'.
```
sh test_iu_xray.sh
```
__3.Performance__  

**IU X-Ray**  
BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDER | METER | ROUGE | SPICE  
---- | ---- | ----| ---- | ---- | ---- | ---- | ----  
49.99 | 31.81 | 22.60 | 16.96 | 33.77 | 19.96 | 39.21 | 31.16 

### Acknowledge  
Some of our codes (i.e., two peer networks) are referring to [cuhksz-nlp/R2Gen](https://github.com/cuhksz-nlp/R2Gen). Thanks for their helpful works.
