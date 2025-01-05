import os
import time

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import json
from randaugment import RandAugment
from PIL import Image
import torch
import cv2
from utils.cutout import SLCutoutPIL
import math


# YOUR_PATH
inte_image_path = './data/sqhy_data/intent_resize'
inte_train_anno_path = './data/intentonomy/intentonomy_train2020.json'
inte_val_anno_path = './data/intentonomy/intentonomy_val2020.json'
inte_test_anno_path = './data/intentonomy/intentonomy_test2020.json'

CLASS_15 = {
    '0': [24, 27],
    '1': [5],
    '2': [13],
    '3': [11],
    '4': [3, 7, 8, 9, 19],
    '5': [4, 6, 20, 21],
    '6': [14],
    '7': [15, 16],
    '8': [2, 22, 23],
    '9': [0],
    '10': [17, 25],
    '11': [10],
    '12': [1],
    '13': [12],
    '14': [18, 26],
}
CLASS_30 = {
    '0': [48, 54],
    '1': [49, 55],
    '2': [10],
    '3': [11],
    '4': [26],
    '5': [27],
    '6': [22],
    '7': [23],
    '8': [6, 14, 16, 18, 38],
    '9': [7, 15, 17, 19, 39],
    '10': [8, 12, 40, 42],
    '11': [9, 13, 41, 43],
    '12': [28],
    '13': [29],
    '14': [30, 32],
    '15': [31, 33],
    '16': [4, 44, 46],
    '17': [5, 45, 47],
    '18': [0],
    '19': [1],
    '20': [34, 50],
    '21': [35, 51],
    '22': [20],
    '23': [21],
    '24': [2],
    '25': [3],
    '26': [24],
    '27': [25],
    '28': [36, 52],
    '29': [37, 53],
}
CLASS_9 = {
    '0': [0, 1],
    '1': [2, 3],
    '2': [4, 5],
    '3': [6],
    '4': [7, 8, 9],
    "5": [10],
    '6': [11],
    '7': [12],
    '8': [13, 14]
}

CLASS_18 = {
    '0': [0, 2],
    '1': [1, 3],
    '2': [4, 6],
    '3': [5, 7],
    '4': [8, 10],
    '5': [9, 11],
    '6': [12],
    '7': [13],
    '8': [14, 16, 18],
    '9': [15, 17, 19],
    '10': [20],
    '11': [21],
    '12': [22],
    '13': [23],
    '14': [24],
    '15': [25],
    '16': [26, 28],
    '17': [27, 29],
}

CLASS_4 = {
    '0': [0,1,3,5,6,13,18],
    '1': [2,7,9,11,12,16],
    '2': [4,8,10,15,17,19],
    '3': [14]
}
CLASS_8 = {
    '0': [0, 2,6,10,12,26,36],
    '1': [1, 3,7,11,13,27,37],
    '2': [4, 14,18,22,24,32],
    '3': [5, 15,19,23,25,33],
    '4': [8, 16,20,30,34,38],
    '5': [9, 17,21,31,35,39],
    '6': [28],
    '7': [29],
}
CLASS_2 = {
    '0': [0,2],
    '1': [1,3],
}
CLASS_2_4 = {
    '0': [0,4],
    '1': [1,5],
    '2': [2,6],
    '3': [3,7]
}


class InteDataSet(data.Dataset):
    def __init__(self, 
                 image_dir, 
                 anno_path, 
                 input_transform=None, 
                 labels_path=None,
    ):
        self.image_dir = image_dir
        self.anno_path = anno_path
        
        self.input_transform = input_transform
        self.labels_path = labels_path
        
        self.labels = []
        if os.path.exists(self.labels_path):
            self.labels = np.load(self.labels_path).astype(np.float64)
            self.labels = (self.labels > 0).astype(np.float64)
            #print(self.labels)
        else:
            print('labels_path not exist, please check the path or run get_label_vector.py first')


    def _load_image(self, index):
        image_path = self._get_image_path(index)
        # print("image_path", image_path)
        if os.path.exists(image_path):
            return Image.open(image_path).convert("RGB")
        else:
            print(index)
            print(image_path)
            print('Data does not exist')
        #if os.path.exists(image_path):
        #    print(index)
        #    return Image.open(image_path).convert("RGB")
        #else:
        #    print(index)
        #    print('Data does not exist')

    #def _get_label(self, index):
    #    image_path = self._get_image_path(index)
    #    with open(self.anno_path, 'r') as f:
    #        annos_dict = json.load(f)
    #        annos_i = annos_dict['annotations'][index]
    #    if os.path.exists(image_path):
    #        return self.labels[index]
    #    else:





    def _get_image_path(self, index):
        with open(self.anno_path, 'r') as f:
            annos_dict = json.load(f)
            annos_i = annos_dict['annotations'][index]
            #id = annos_i['id']
            #if id != index:
            #    raise ValueError('id not equal to index')
            img_id_i = annos_i['image_id']
            
            imgs = annos_dict['images']

            for img in imgs:
                if img['id'] == img_id_i:
                    image_file_name = img['filename']
                    image_file_path = os.path.join(self.image_dir, image_file_name)
                    break
        
        return image_file_path
                    
    def __getitem__(self, index):
        input = self._load_image(index)


        resize = transforms.Resize((224, 224))
        input = resize(input)
        time1 = time.time()
        # # 模糊
        # # 将图片灰度标准化
        # img = np.array(input)
        # img = img / 255
        # # 产生高斯 noise
        # time2 = time.time()
        # # print('1', time2-time1)
        # noise = np.random.normal(0, 0.05, size=img.shape)
        # time3 = time.time()
        # # print('2', time3 - time2)
        # # 将噪声和图片叠加a
        # gaussian_out = img + noise
        # # 将超过 1 的置 1，低于 0 的置 0
        # gaussian_out = np.clip(gaussian_out, 0, 1)
        # # 将图片灰度范围的恢复为 0-255
        # input = np.uint8(gaussian_out * 255)
        # input = Image.fromarray(input)

        # #motion
        # degree = 30
        # angel = 360
        # resize = transforms.Resize((224, 224))
        # input = resize(input)
        # img = np.array(input)
        # M = cv2.getRotationMatrix2D((degree/2,degree/2),angel,1)
        # motion_blur_kernel = np.diag(np.ones(degree))
        # motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        # motion_blur_kernel = motion_blur_kernel/degree
        #
        # blurred = cv2.filter2D(img, -1, motion_blur_kernel)
        #
        # cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        # blurred = np.array(blurred, dtype=np.uint8)
        # input = Image.fromarray(blurred)

        # #AddHaze1
        # img = np.array(input)
        # img_f = img/255.0
        # (row, col, chs) = img.shape
        # A = 0.8
        # beta = 0.5
        # size = math.sqrt(max(row, col))
        # center = (row // 2, col // 2)
        # for j in range(row):
        #     for l in range(col):
        #         d = -0.04 * math.sqrt((j - center[0]) ** 2 +(l - center[1]) ** 2) + size
        #         td = math.exp(-beta * d)
        #         img_f[j][l][:] * td +A * (1 - td)
        # input = Image.fromarray((np.uint8(img_f * 255)))



        if self.input_transform:
            input = self.input_transform(input)
        label = self.labels[index]
        return input, label
    
    def __len__(self):
        return self.labels.shape[0]


def get_datasets(args):
    if args.orid_norm:
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])
        # print("mean=[0, 0, 0], std=[1, 1, 1]")
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        # print("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

    train_data_transform_list = [transforms.Resize((args.img_size, args.img_size)),
                                            RandAugment(),
                                               transforms.ToTensor(),
                                               normalize]
    if args.cutout:
        print("Using Cutout!!!")
        train_data_transform_list.insert(1, SLCutoutPIL(n_holes=args.n_holes, length=args.length))
        
    train_data_transform = transforms.Compose(train_data_transform_list)
    test_data_transform = transforms.Compose([
                                            transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor(),
                                            normalize])
    
    if args.dataname == 'intentonomy':
        # ! config your data path here.
        dataset_dir = args.dataset_dir
        train_dataset = InteDataSet(
            image_dir=inte_image_path,
            anno_path=inte_train_anno_path,
            input_transform=train_data_transform,
            labels_path='/home/zhouyifan/test/HLEG-main/data/intentonomy/train_label_vectors_intentonomy2020.npy',
        )
        val_dataset = InteDataSet(
            image_dir=inte_image_path,
            anno_path=inte_val_anno_path,
            input_transform=test_data_transform,
            labels_path='/home/zhouyifan/test/HLEG-main/data/intentonomy/val_label_vectors_intentonomy2020.npy',
        )
        test_dataset = InteDataSet(
            image_dir=inte_image_path,
            anno_path=inte_test_anno_path,
            input_transform=test_data_transform,
            labels_path='/home/zhouyifan/test/HLEG-main/data/intentonomy/test_label_vectors_intentonomy2020.npy',
        )

    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)

    print("len(train_dataset):", len(train_dataset)) 
    print("len(val_dataset):", len(val_dataset))
    print("len(test_dataset):", len(test_dataset))
    return train_dataset, val_dataset, test_dataset







