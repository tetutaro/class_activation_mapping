#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class GradCAM(BaseCAM):
    """Grad-CAM

    "Grad-CAM:
    Visual Explanations from Deep Networks via Gradient-based Localization"

    https://arxiv.org/abs/1610.02391

    Args:
        backbone (Backbone): resouce of CNN.
    """

    def __init__(
        self: GradCAM,
        backbone: Backbone,
    ) -> None:
        super().__init__(
            name="Grad-CAM",
            backbone=backbone,
            activation_weight="gradient",
        )
        return
