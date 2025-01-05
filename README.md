# UCFN: Uncertainty-aware Cross-granularity Fusion Network for Visual Intention Understanding
## dataset

[DATA.md](https://github.com/KMnP/intentonomy/blob/master/DATA.md)

After downloading the dataset file, place it in the 'data' folder.
```
inte_image_path = './data/sqhy_data/intent_resize'
inte_train_anno_path = './data/intentonomy/intentonomy_train2020.json'
inte_val_anno_path = './data/intentonomy/intentonomy_val2020.json'
inte_test_anno_path = './data/intentonomy/intentonomy_test2020.json'
```
>NOTE: Please refer to our **edit_numpy.py** file for the data processing code.

## backbone
TResNetM pretrained on ImageNet 21k is available at [TResNetM_pretrained_model](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ASL/MS_COCO_TRresNet_M_224_81.8.pth)


After downloading the file, place it in the current folder as './tresnet_m_224_21k.pth'.
>NOTE: This framework should work for the previous versions: **timm ==0.5.4 inplace_abn=1.1.0**



## checkpoint
Please download the checkpoint [checkpoint](https://drive.google.com/file/d/1IaH8L3dIso4MOcHR4heEkZ2Xyz4noD9b/view?usp=drive_link) and place it in the 'checkpoint' folder as './checkpoint/checkpoint.tar'.


## usage
```sh
# training
python main.py 
# evaluation
python main.py -e
```
## File Structure
```
├── UCFN
    ├── checkpoint
        ├── checkpoint.tar
    ├── data
        ├── intentonomy
        ├── sqhy_data
    ├── data_utils
    ├── models
    ├── outputs
    ├── utils
    ├── main.py
    ├── readme.md
    ├── tresnet_m_224_21k.pth
```
