#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""wrapper classes, functions of torchvision.models.vgg"""
from __future__ import annotations
from typing import Optional, Any

import torch
from torch import Tensor
from torchvision.models import (
    VGG,
    VGG11_Weights,
    VGG13_Weights,
    VGG16_Weights,
    VGG19_Weights,
)
from torchvision.models._api import register_model, WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.vgg import cfgs, make_layers

from cam.backbones.backbone import Backbone


class wVGG(VGG):
    def forward_classifier(self: VGG, activation: Tensor) -> Tensor:
        """forward the activation of the last Conv. Layer
        to the classifier block of the CNN model.
        """
        for layer in self.features[-2][1:]:
            activation = layer(activation)
        return self.classifier(
            torch.flatten(
                self.avgpool(
                    self.features[-1](activation),
                ),
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


@register_model()
def w_vgg11(
    *,
    weights: Optional[VGG16_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> wVGG:
    """wrapper function of torchvision.models.vgg.vgg11"""
    weights = VGG11_Weights.verify(weights)
    return _w_vgg("A", False, weights, progress, **kwargs)


@register_model()
def w_vgg13(
    *,
    weights: Optional[VGG13_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> wVGG:
    """wrapper function of torchvision.models.vgg.vgg13"""
    weights = VGG13_Weights.verify(weights)
    return _w_vgg("B", False, weights, progress, **kwargs)


@register_model()
def w_vgg16(
    *,
    weights: Optional[VGG16_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> wVGG:
    """wrapper function of torchvision.models.vgg.vgg16"""
    weights = VGG16_Weights.verify(weights)
    return _w_vgg("D", False, weights, progress, **kwargs)


@register_model()
def w_vgg19(
    *,
    weights: Optional[VGG19_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> wVGG:
    """wrapper function of torchvision.models.vgg.vgg19"""
    weights = VGG19_Weights.verify(weights)
    return _w_vgg("E", False, weights, progress, **kwargs)


backbone_vgg11: Backbone = Backbone(
    net=w_vgg11,
    weights=VGG11_Weights.IMAGENET1K_V1,
)


backbone_vgg13: Backbone = Backbone(
    net=w_vgg13,
    weights=VGG13_Weights.IMAGENET1K_V1,
)


backbone_vgg16: Backbone = Backbone(
    net=w_vgg16,
    weights=VGG16_Weights.IMAGENET1K_V1,
)


backbone_vgg19: Backbone = Backbone(
    net=w_vgg19,
    weights=VGG19_Weights.IMAGENET1K_V1,
)
