#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from cam.backbones.mobilenet_v3 import (
    backbone_mobilenet_v3_small,
    backbone_mobilenet_v3_large,
)
from cam.backbones.vgg import (
    backbone_vgg11,
    backbone_vgg13,
    backbone_vgg16,
    backbone_vgg19,
)
from cam.backbones.resnet import (
    backbone_resnet18,
    backbone_resnet34,
    backbone_resnet50,
    backbone_resnet101,
    backbone_resnet152,
    backbone_resnext50_32x4d,
    backbone_resnext101_32x8d,
    backbone_resnext101_64x4d,
    backbone_wide_resnet50_2,
    backbone_wide_resnet101_2,
)

__all__ = [
    "backbone_mobilenet_v3_small",
    "backbone_mobilenet_v3_large",
    "backbone_vgg11",
    "backbone_vgg13",
    "backbone_vgg16",
    "backbone_vgg19",
    "backbone_resnet18",
    "backbone_resnet34",
    "backbone_resnet50",
    "backbone_resnet101",
    "backbone_resnet152",
    "backbone_resnext50_32x4d",
    "backbone_resnext101_32x8d",
    "backbone_resnext101_64x4d",
    "backbone_wide_resnet50_2",
    "backbone_wide_resnet101_2",
]
