#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Any

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class EigenGradCAM(BaseCAM):
    """EigenGrad-CAM

    Eigen-CAM + Grad-CAM

    Args:
        backbone (Backbone): resouce of CNN.
    """

    def __init__(
        self: EigenGradCAM,
        backbone: Backbone,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="EigenGrad-CAM",
            backbone=backbone,
            activation_weight="gradient",
            channel_weight="eigen",
        )
        return
