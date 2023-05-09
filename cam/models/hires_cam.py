#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Any

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class HiResCAM(BaseCAM):
    """HiRes-CAM

    "Use HiResCAM instead of Grad-CAM
    for faithful explanations of convolutional neural networks"

    https://arxiv.org/abs/2011.08891

    Args:
        backbone (Backbone): resouce of CNN.
    """

    def __init__(
        self: HiResCAM,
        backbone: Backbone,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="HiRes-CAM",
            backbone=backbone,
            activation_weight="gradient",
            gradient_gap=False,
        )
        return
