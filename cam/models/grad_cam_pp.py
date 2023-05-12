#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Any

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class GradCAMpp(BaseCAM):
    """Grad-CAM++

    A. Chattopadhyay, et al.
    "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks"
    WACV 2018.

    https://arxiv.org/abs/1710.11063

    Args:
        backbone (Backbone): resouce of CNN.
    """

    def __init__(
        self: GradCAMpp,
        backbone: Backbone,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="Grad-CAM++",
            backbone=backbone,
            activation_weight="gradient++",
        )
        return
