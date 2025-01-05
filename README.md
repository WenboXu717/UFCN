## dataset

[DATA.md](https://github.com/KMnP/intentonomy/blob/master/DATA.md)

>NOTE: Please refer to our **edit_numpy.py** file for the data processing code.

## backbone
TResNetM pretrained on ImageNet 21k is available at [TResNetM_pretrained_model](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ASL/MS_COCO_TRresNet_M_224_81.8.pth)

>NOTE: This framework should work for the previous versions: **timm ==0.5.4 inplace_abn=1.1.0**

## usage
```sh
# training
python main.py 
# evaluation
python main.py -e
```

