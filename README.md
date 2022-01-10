# Multi-modal-medical-image-caption

This is the implementation of [Multi-modal Multi-label Medical Image Caption with Semantic Consistency].

## Requirements

- `torch==1.8.1`
- `torchvision==0.9.1`
- `opencv-python==4.5.2.52`


## Download R2Gen Models
You can download the models we trained for IU X-Ray dataset.

The pre-trained R2Gen models. You can download the models and run them on the corresponding datasets to replicate our results.

| Section   | BaiduNetDisk                                                 | GoogleDrive                                                  | Description                          |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------ |
| IU X-Ray  | [download](https://pan.baidu.com/s/1JEFepQt4OVQvMrUwnK3low) (Password: mqjc) | [download](https://drive.google.com/file/d/1fQXpf4vz5t2QYQ89iRXv0U_OVA0MNCtJ/view?usp=sharing) | R2Gen model trained on **IU X-Ray**  |
## Datasets
We use a public dataset (IU X-Ray) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.

## Run on IU X-Ray

Run `bash run_iu_xray.sh` to train a model on the IU X-Ray data.
