#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""wrapper classes, functions of torchvision.models.resnet"""
from __future__ import annotations
from typing import Type, List, DefaultDict, Union, Optional, Any
from collections import defaultdict

from torch import Tensor
import torch.nn as nn
from torchvision.models import (
    ResNet,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    ResNeXt50_32X4D_Weights,
    ResNeXt101_32X8D_Weights,
    ResNeXt101_64X4D_Weights,
    Wide_ResNet50_2_Weights,
    Wide_ResNet101_2_Weights,
)
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.resnet import BasicBlock, Bottleneck

from cam.backbones.backbone import Backbone


class wResNet(ResNet):
    """the wrapper class of torchvision.models.resnet.ResNet."""

    def get_all_conv_layers(self: wResNet) -> DefaultDict[int, List[str]]:
        """get names of all Conv. Layers.

        Returns:
            DefaultDict[int, List[str]]: block number -> names of Conv. Layers.
        """
        self.relu.inplace = False
        all_conv_layers: DefaultDict[List] = defaultdict(list)
        for i, layer in enumerate(
            [self.layer1, self.layer2, self.layer3, self.layer4]
        ):
            block: int = i + 1
            name: str = f"layer{block}"
            for module in layer.named_modules():
                if isinstance(module[1], nn.ReLU):
                    # https://github.com/frgfm/torch-cam/issues/72
                    module[1].inplace = False
                    continue
                if isinstance(module[1], nn.Conv2d):
                    all_conv_layers[block].append(f"{name}." + module[0])
        return all_conv_layers

    def get_avgpool_size(self: wResNet) -> int:
        """calc the output size of the avgpool part.

        Returns:
            int: the size of the avgpool part.
        """
        avgpool_size: int = 1
        for size in self.avgpool.output_size:
            avgpool_size *= size
        return avgpool_size

    def get_class_weight(self: wResNet) -> Tensor:
        """get weight of classifier.

        Returns:
            Tensor: the weight of classifier.
        """
        for param in self.fc.named_parameters():
            if param[0] == "weight":
                return param[1].data.clone().detach()

    def forward_classifier(self: wResNet, activation: Tensor) -> Tensor:
        """because of shortcut connections in the Residual block,
        it is impossible to forward activation just after the Conv. Layer.
        """
        raise NotImplementedError("Resnet doesn't support Ablation-CAM")


def _w_resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> wResNet:
    """wrapper function of torchvision.models.resnet._resnet"""
    if weights is not None:
        _ovewrite_named_param(
            kwargs, "num_classes", len(weights.meta["categories"])
        )
    model = wResNet(block, layers, **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    return model


def w_resnet18(
    *,
    weights: Optional[ResNet18_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> wResNet:
    """wrapper function of torchvision.models.resnet.resnet18"""
    weights = ResNet18_Weights.verify(weights)
    return _w_resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


def w_resnet34(
    *,
    weights: Optional[ResNet34_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> wResNet:
    """wrapper function of torchvision.models.resnet.resnet34"""
    weights = ResNet34_Weights.verify(weights)
    return _w_resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)


def w_resnet50(
    *,
    weights: Optional[ResNet50_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> wResNet:
    """wrapper function of torchvision.models.resnet.resnet34"""
    weights = ResNet50_Weights.verify(weights)
    return _w_resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


def w_resnet101(
    *,
    weights: Optional[ResNet101_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> wResNet:
    """wrapper function of torchvision.models.resnet.resnet101"""
    weights = ResNet101_Weights.verify(weights)
    return _w_resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


def w_resnet152(
    *,
    weights: Optional[ResNet152_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> wResNet:
    """wrapper function of torchvision.models.resnet.resnet152"""
    weights = ResNet152_Weights.verify(weights)
    return _w_resnet(Bottleneck, [3, 8, 36, 3], weights, progress, **kwargs)


def w_resnext50_32x4d(
    *,
    weights: Optional[ResNeXt50_32X4D_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> wResNet:
    """wrapper function of torchvision.models.resnet.resnet50_32x4d"""
    weights = ResNeXt50_32X4D_Weights.verify(weights)
    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _w_resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


def w_resnext101_32x8d(
    *,
    weights: Optional[ResNeXt101_32X8D_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> wResNet:
    weights = ResNeXt101_32X8D_Weights.verify(weights)
    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 8)
    return _w_resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


def w_resnext101_64x4d(
    *,
    weights: Optional[ResNeXt101_64X4D_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> wResNet:
    """wrapper function of torchvision.models.resnet.resnet101_64x4d"""
    weights = ResNeXt101_64X4D_Weights.verify(weights)
    _ovewrite_named_param(kwargs, "groups", 64)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _w_resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


def w_wide_resnet50_2(
    *,
    weights: Optional[Wide_ResNet50_2_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> wResNet:
    """wrapper function of torchvision.models.resnet.wide_resnet50_2"""
    weights = Wide_ResNet50_2_Weights.verify(weights)
    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _w_resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


def w_wide_resnet101_2(
    *,
    weights: Optional[Wide_ResNet101_2_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> wResNet:
    """wrapper function of torchvision.models.resnet.wide_resnet101_2"""
    weights = Wide_ResNet101_2_Weights.verify(weights)
    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _w_resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


backbone_resnet18: Backbone = Backbone(
    cnn_name="ResNet18",
    net=w_resnet18,
    weights=ResNet18_Weights.IMAGENET1K_V1,
)
backbone_resnet34: Backbone = Backbone(
    cnn_name="ResNet34",
    net=w_resnet34,
    weights=ResNet34_Weights.IMAGENET1K_V1,
)
backbone_resnet50: Backbone = Backbone(
    cnn_name="ResNet50",
    net=w_resnet50,
    weights=ResNet50_Weights.IMAGENET1K_V2,
)
backbone_resnet101: Backbone = Backbone(
    cnn_name="ResNet101",
    net=w_resnet101,
    weights=ResNet101_Weights.IMAGENET1K_V2,
)
backbone_resnet152: Backbone = Backbone(
    cnn_name="ResNet152",
    net=w_resnet152,
    weights=ResNet152_Weights.IMAGENET1K_V2,
)
backbone_resnext50_32x4d: Backbone = Backbone(
    cnn_name="ResNeXt50_32X4D",
    net=w_resnext50_32x4d,
    weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2,
)
backbone_resnext101_32x8d: Backbone = Backbone(
    cnn_name="ResNeXt101_32X8D",
    net=w_resnext101_32x8d,
    weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2,
)
backbone_resnext101_64x4d: Backbone = Backbone(
    cnn_name="ResNeXt101_64X4D",
    net=w_resnext101_64x4d,
    weights=ResNeXt101_64X4D_Weights.IMAGENET1K_V1,
)
backbone_wide_resnet50_2: Backbone = Backbone(
    cnn_name="WideResNet50_2",
    net=w_wide_resnet50_2,
    weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2,
)
backbone_wide_resnet101_2: Backbone = Backbone(
    cnn_name="WideResNet101_2",
    net=w_wide_resnet101_2,
    weights=Wide_ResNet101_2_Weights.IMAGENET1K_V2,
)
