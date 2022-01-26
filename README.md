# Multi-modal-medical-image-caption

This is the implementation of [Multi-modal Multi-label Medical Image Caption with Semantic Consistency].

## Requirements

- `torch==1.8.1`
- `torchvision==0.9.1`
- `opencv-python==4.5.2.52`


## Datasets
We use a public dataset (IU X-Ray) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/drive/folders/186KDV48o-jtK09b4yzNHbfo-kQDUyEsw?usp=sharing) and then put the files in `data/iu_xray`.

## Run on IU X-Ray

Run `bash run_iu_xray.sh` to train a model on the IU X-Ray data.
