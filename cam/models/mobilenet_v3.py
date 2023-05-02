#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""wrapper class, functions of torchvision.models.mobilenet_v3"""
from __future__ import annotations
from typing import List, Optional, Any

import torch
from torch import Tensor
from torchvision.models import MobileNetV3, MobileNet_V3_Large_Weights
from torchvision.models._api import register_model, WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.mobilenetv3 import _mobilenet_v3_conf

from cam.libs_cam import ResourceCNN


class wMobileNetV3(MobileNetV3):
    """the wrapper class of MobileNetV3.

    implement forward_classifier function
    to be anable to forward the activation
    of the last Conv. Layer of MobileNetV3
    to the classifier block of MobileNetV3 only.
    """

    def forward_classifier(self: wMobileNetV3, activation: Tensor) -> Tensor:
        """forward the activation of the last Conv. Layer
        to the classifier block of the CNN model.
        """
        return self.classifier(
            torch.flatten(
                self.avgpool(
                    self.features[16][2](self.features[16][1](activation)),
                ),
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
    """wrapper function of torchvision.models.mobilenetv3._mobilente_v3()"""
    if weights is not None:
        _ovewrite_named_param(
            kwargs, "num_classes", len(weights.meta["categories"])
        )
    model = wMobileNetV3(inverted_residual_setting, last_channel, **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    return model


@register_model()
def w_mobilenet_v3_large(
    *,
    weights: Optional[MobileNet_V3_Large_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> wMobileNetV3:
    """wrapper function of torchvision.models.mobilenetv3.mobilenet_v3_large"""
    weights = MobileNet_V3_Large_Weights.verify(weights)
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(
        "mobilenet_v3_large"
    )
    return _w_mobilenet_v3(
        inverted_residual_setting,
        last_channel,
        weights,
        progress,
        **kwargs,
    )


resource_mobilenet_v3: ResourceCNN = ResourceCNN(
    net=w_mobilenet_v3_large,
    weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2,
)
