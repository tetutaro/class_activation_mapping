#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""wrapper classes, functions of torchvision.models.mobilenet_v3"""
from __future__ import annotations
from typing import List, Optional, DefaultDict, Any
from collections import defaultdict

import torch
from torch import Tensor
import torch.nn as nn
from torchvision.models import (
    MobileNetV3,
    MobileNet_V3_Small_Weights,
    MobileNet_V3_Large_Weights,
)
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.mobilenetv3 import _mobilenet_v3_conf

from cam.backbones.backbone import Backbone


class wMobileNetV3(MobileNetV3):
    """the wrapper class of
    torchvision.models.mobilenetv3.MobileNetV3.
    """

    def get_all_conv_layers(self: wMobileNetV3) -> DefaultDict[int, List[str]]:
        """get names of all Conv. Layers.

        Returns:
            DefaultDict[int, List[str]]: block number -> names of Conv. Layers.
        """
        all_conv_layers: DefaultDict[List] = defaultdict(list)
        for module in self.features.named_modules():
            if isinstance(module[1], nn.ReLU):
                # https://github.com/frgfm/torch-cam/issues/72
                module[1].inplace = False
                continue
            if isinstance(module[1], nn.Conv2d):
                block: int = int(module[0].split(".")[0])
                if block == 0:
                    # ignore the first block
                    # because the first block just expands color information
                    # to following channels
                    continue
                all_conv_layers[block].append("features." + module[0])
        return all_conv_layers

    def get_avgpool_size(self: wMobileNetV3) -> int:
        """calc the output size of the avgpool part.

        Returns:
            int: the size of the avgpool part.
        """
        return self.avgpool.output_size

    def get_class_weight(self: wMobileNetV3) -> Tensor:
        """get weight of classifier.

        Returns:
            Tensor: the weight of classifier.
        """
        cweight: Optional[Tensor] = None
        for module in self.classifier.named_modules():
            if isinstance(module[1], nn.modules.linear.Linear):
                for param in module[1].named_parameters():
                    if param[0] == "weight":
                        weight: Tensor = param[1].data.clone().detach()
                        if cweight is None:
                            cweight = weight
                        else:
                            cweight = weight @ cweight
        return cweight

    def forward_classifier(self: wMobileNetV3, activation: Tensor) -> Tensor:
        """forward the activation of the last Conv. Layer
        to the classifier block of the CNN model.

        Args:
            activation (Tensor): activation

        Returns:
            Tensor: score
        """
        for offset in range(1, len(self.features[-1])):
            activation = self.features[-1][offset](activation)
        return self.classifier(
            torch.flatten(
                self.avgpool(activation),
                1,
            )
        )


def _w_mobilenet_v3(
    inverted_residual_setting: List,
    last_channel: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> wMobileNetV3:
    """wrapper function of
    torchvision.models.mobilenetv3._mobilente_v3()
    """
    if weights is not None:
        _ovewrite_named_param(
            kwargs, "num_classes", len(weights.meta["categories"])
        )
    model = wMobileNetV3(inverted_residual_setting, last_channel, **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    return model


def w_mobilenet_v3_small(
    *,
    weights: Optional[MobileNet_V3_Small_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> MobileNetV3:
    """wrapper function of
    torchvision.models.mobilenetv3.mobilenet_v3_small()
    """
    weights = MobileNet_V3_Small_Weights.verify(weights)
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(
        "mobilenet_v3_small", **kwargs
    )
    return _w_mobilenet_v3(
        inverted_residual_setting, last_channel, weights, progress, **kwargs
    )


def w_mobilenet_v3_large(
    *,
    weights: Optional[MobileNet_V3_Large_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> wMobileNetV3:
    """wrapper function of
    torchvision.models.mobilenetv3.mobilenet_v3_large()
    """
    weights = MobileNet_V3_Large_Weights.verify(weights)
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(
        "mobilenet_v3_large", **kwargs
    )
    return _w_mobilenet_v3(
        inverted_residual_setting, last_channel, weights, progress, **kwargs
    )


backbone_mobilenet_v3_small: Backbone = Backbone(
    cnn_name="MobileNetV3Small",
    net=w_mobilenet_v3_small,
    weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1,
)
backbone_mobilenet_v3_large: Backbone = Backbone(
    cnn_name="MobileNetV3Large",
    net=w_mobilenet_v3_large,
    weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2,
)
