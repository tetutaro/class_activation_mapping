#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Any

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class PolyCAM(BaseCAM):
    """Poly-CAM

    A. Englebert, et al.
    "Poly-CAM:
    High resolution class activation map for convolutional neural networks"
    ICPR 2022.

    https://arxiv.org/abs/2204.13359

    Args:
        backbone (Backbone): resouce of CNN.
    """

    def __init__(
        self: PolyCAM,
        backbone: Backbone,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="Poly-CAM",
            backbone=backbone,
            activation_weight="gradient",
            gradient_gap=False,
            high_resolution=True,
        )
        return
