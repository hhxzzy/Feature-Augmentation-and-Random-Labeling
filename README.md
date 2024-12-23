## Introduction

This zip file contains the Pytorch implementation of our paper **Regularizing Classifier For Few-Shot Object Detection via Feature Augmentation and Random Labeling**. Our code is based on the baseline algorithm [DeFRCN](https://github.com/er-muyue/DeFRCN/tree/main). 

The methods of our paper are mainly implemented in `FARL/defrcn/modeling/roi_heads/roi_heads.py`.

## Quick Start

**1. Prepare Requirements**

* Create a virtual environment

* Install PyTorch 1.6.0 with CUDA 10.1 (other versions may also work)

  ```shell
  pip3 install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
  ```

* Install Detectron2
  ```shell
  python3 -m pip install detectron2==0.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
  ```
* Install other requirements. 
  ```shell
  python3 -m pip install -r requirements.txt
  ```

**2. Prepare Data and Weights**

* Data Preparation

  According to our baseline algorithm DeFRCN's data preparation, we can download the datasets using the following links:

  |  Dataset  | Size |                         GoogleDrive                          |                             Note                             |
  | :-------: | :--: | :----------------------------------------------------------: | :----------------------------------------------------------: |
  |  VOC2007  | 0.8G | [download](https://drive.google.com/file/d/1BcuJ9j9Mtymp56qGSOfYxlXN4uEVyxFm/view?usp=sharing) |                              -                               |
  |  VOC2012  | 3.5G | [download](https://drive.google.com/file/d/1NjztPltqm-Z-pG94a6PiPVP4BgD8Sz1H/view?usp=sharing) |                              -                               |
  | vocsplit  | <1M  | [download](https://drive.google.com/file/d/1BpDDqJ0p-fQAFN_pthn2gqiK5nWGJ-1a/view?usp=sharing) | refer from [TFA](https://github.com/ucbdrive/few-shot-object-detection#models) |
  |   COCO    | ~19G |                              -                               |  download from [offical](https://cocodataset.org/#download)  |
  | cocosplit | 174M | [download](https://drive.google.com/file/d/1T_cYLxNqYlbnFNJt8IVvT7ZkWb5c0esj/view?usp=sharing) | refer from [TFA](https://github.com/ucbdrive/few-shot-object-detection#models) |

  Unzip the downloaded datasets and put them into project directory as follows:

  ```angular2html
  FARL
    | -- ...
    | -- datasets
    |     | -- coco (trainval2014/*.jpg, val2014/*.jpg, annotations/*.json)
    |     | -- cocosplit
    |     | -- VOC2007
    |     | -- VOC2012
    |     | -- vocsplit
    | -- ...
  ```

* Weights Preparation

  Our baseline algorithm DeFRCN designs a PCB module to improve the classification performance, which utilize a ImageNet pretrained model. The ImageNet pretrained model can be downloaded from [GoogleDrive](https://drive.google.com/file/d/1rsE20_fSkYeIhFaNU04rBfEDkMENLibj/view?usp=sharing). Unzip the file, and the model named `resnet101-5d3b4d8f.pth` is for PCB module.

  Our paper targets the fine-tuning stage, so we need to download the weight after base training from our baseline algorithm, and use it as initial weight for the fine-tuning process. As for DeFRCN, we can download their results from [GoogleDrive](https://drive.google.com/file/d/1Ff5jP4PCDDPQ7lzsageZsauFWer73QIl/view?usp=sharing) and [GoogleDrive](https://drive.google.com/file/d/1WUM2X-pPzox2fQz4aLi3YzxGgscpnoHU/view?usp=sharing) for PASCAL VOC and MSCOCO respectively. Unzip the files and rename the weights as follows:

  | Path                                                         | Rename                             | Note                                      |
  | ------------------------------------------------------------ | ---------------------------------- | ----------------------------------------- |
  | voc/defrcn_one/defrcn_det_r101_base1/model_reset_surgery.pth | voc_split1_model_reset_surgery.pth | weight after base training for VOC split1 |
  | voc/defrcn_one/defrcn_det_r101_base2/model_reset_surgery.pth | voc_split2_model_reset_surgery.pth | weight after base training for VOC split2 |
  | voc/defrcn_one/defrcn_det_r101_base3/model_reset_surgery.pth | voc_split3_model_reset_surgery.pth | weight after base training for VOC split3 |
  | coco/defrcn_one/defrcn_det_r101_base/model_reset_surgery.pth | coco_model_reset_surgery.pth       | weight after base training for COCO       |

  Put the pretrained weights into project directory as follows:

  ```angular2html
  FARL
    | -- ...
    | -- pretrained
    |     | -- resnet101-5d3b4d8f.pth
    |     | -- voc_split1_model_reset_surgery.pth
    |     | -- voc_split2_model_reset_surgery.pth
    |     | -- voc_split3_model_reset_surgery.pth
    |     | -- coco_model_reset_surgery.pth
    | -- ...
  ```

**3. Train and Evaluate**

* PASCAL VOC

  ```shell
  bash run_voc.sh EXP_NAME SPLIT_ID (1, 2 or 3) METHOD (baseline, FA, FARL or iFARL)
  ```

* MSCOCO

  ```shell
  bash run_coco.sh EXP_NAME METHOD (baseline, FA, FARL or iFARL)
  ```

  

