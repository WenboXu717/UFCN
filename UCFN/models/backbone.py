# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
We borrow the positional encoding from Detr and adding some other backbones.
"""
from collections import OrderedDict
import os
import warnings
import numpy as np

import torch
from torch.functional import Tensor
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

import models
from utils.misc import clean_state_dict
from .position_encoding import build_position_encoding

from models.tresnet.tresnet import TResnetM, TResnetL, TResnetXL

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, args=None):
        super().__init__(backbone, position_embedding)
        # self.args = args
        if args is not None and 'interpotaion' in vars(args) and args.interpotaion:
            self.interpotaion = True
        else:
            self.interpotaion = False


    def forward(self, input: Tensor):
        xs = self[0](input)
        out: List[Tensor] = []
        pos = []
        if isinstance(xs, dict):
            for name, x in xs.items():
                out.append(x)
                # position encoding
                pos.append(self[1](x).to(x.dtype))
        else:
            # for swin Transformer
            out.append(xs)
            pos.append(self[1](xs).to(xs.dtype))
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    backbone = get_model_tresnet()
    bb_num_channels = 2048
    model = Joiner(backbone, position_embedding, args)
    model.num_channels = bb_num_channels
    return model

def get_model_tresnet():
        """
        Create model tresnet
        Load Checkpoint from resume or pretrained weight
        """
        # YOUR_PATH/backbone
        pretrained_path = "./tresnet_m_224_21k.pth"
        num_classes = 56
        model_name = 'tresnet_m'
        model = my_create_model(model_name, num_classes)
        state = torch.load(pretrained_path, map_location='cpu')
        state = {(k if 'body.' not in k else k[5:]): v for k, v in state['model'].items()}
        filtered_dict = {k: v for k, v in state.items() if
                        (k in model.state_dict() and 'head.fc' not in k)}

        model.load_state_dict(filtered_dict, strict=False)
        model = nn.Sequential(
            model.space_to_depth,
            model.conv1,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

        return model

def my_create_model(model_name, num_classes):
    """Create a model, with model_name and num_classes
    """
    model_params = {'num_classes': num_classes}

    if model_name=='tresnet_m':
        model = TResnetM(model_params)
    elif model_name=='tresnet_l':
        model = TResnetL(model_params)
    elif model_name=='tresnet_xl':
        model = TResnetXL(model_params)
    
    else:
        print("model: {} not found !!".format(model_name))
        exit(-1)

    return model
