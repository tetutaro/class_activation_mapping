#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""wrapper classes, functions of torchvision.models.vgg"""
from __future__ import annotations
from typing import List, Optional, DefaultDict, Any
from collections import defaultdict

import torch
from torch import Tensor
import torch.nn as nn
from torchvision.models import (
    VGG,
    VGG11_Weights,
    VGG13_Weights,
    VGG16_Weights,
    VGG19_Weights,
)
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.vgg import cfgs, make_layers

from cam.backbones.backbone import Backbone


class wVGG(VGG):
    """the wrapper class of torchvision.models.vgg.VGG."""

    def get_all_conv_layers(self: wVGG) -> DefaultDict[int, List[str]]:
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

    def get_avgpool_size(self: wVGG) -> int:
        """calc the output size of the avgpool part.

        Returns:
            int: the size of the avgpool part.
        """
        avgpool_size: int = 1
        for size in self.avgpool.output_size:
            avgpool_size *= size
        return avgpool_size

    def get_class_weight(self: wVGG) -> Tensor:
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

    def forward_classifier(self: VGG, activation: Tensor) -> Tensor:
        """forward the activation of the last Conv. Layer
        to the classifier block of the CNN model.

        Args:
            activation (Tensor): activation

        Returns:
            Tensor: score
        """
        for layer in self.features[-2:]:
            activation = layer(activation)
        return self.classifier(
            torch.flatten(
                self.avgpool(activation),
                1,
            )
        )


def _w_vgg(
    cfg: str,
    batch_norm: bool,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> wVGG:
    """wrapper function of torchvision.models.vgg._vgg()"""
    if weights is not None:
        kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            _ovewrite_named_param(
                kwargs, "num_classes", len(weights.meta["categories"])
            )
    model = wVGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    return model


def w_vgg11(
    *,
    weights: Optional[VGG16_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> wVGG:
    """wrapper function of torchvision.models.vgg.vgg11()"""
    weights = VGG11_Weights.verify(weights)
    return _w_vgg("A", False, weights, progress, **kwargs)


def w_vgg13(
    *,
    weights: Optional[VGG13_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> wVGG:
    """wrapper function of torchvision.models.vgg.vgg13()"""
    weights = VGG13_Weights.verify(weights)
    return _w_vgg("B", False, weights, progress, **kwargs)


def w_vgg16(
    *,
    weights: Optional[VGG16_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> wVGG:
    """wrapper function of torchvision.models.vgg.vgg16()"""
    weights = VGG16_Weights.verify(weights)
    return _w_vgg("D", False, weights, progress, **kwargs)


def w_vgg19(
    *,
    weights: Optional[VGG19_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> wVGG:
    """wrapper function of torchvision.models.vgg.vgg19()"""
    weights = VGG19_Weights.verify(weights)
    return _w_vgg("E", False, weights, progress, **kwargs)


backbone_vgg11: Backbone = Backbone(
    cnn_name="VGG11",
    net=w_vgg11,
    weights=VGG11_Weights.IMAGENET1K_V1,
)
backbone_vgg13: Backbone = Backbone(
    cnn_name="VGG13",
    net=w_vgg13,
    weights=VGG13_Weights.IMAGENET1K_V1,
)
backbone_vgg16: Backbone = Backbone(
    cnn_name="VGG16",
    net=w_vgg16,
    weights=VGG16_Weights.IMAGENET1K_V1,
)
backbone_vgg19: Backbone = Backbone(
    cnn_name="VGG19",
    net=w_vgg19,
    weights=VGG19_Weights.IMAGENET1K_V1,
)
