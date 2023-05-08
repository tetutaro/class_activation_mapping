#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class XGradCAM(BaseCAM):
    """XGrad-CAM

    "Axiom-based Grad-CAM:
    Towards Accurate Visualization and Explanation of CNNs"

    https://arxiv.org/abs/2008.02312

    Args:
        backbone (Backbone): resouce of CNN.
    """

    def __init__(
        self: XGradCAM,
        backbone: Backbone,
    ) -> None:
        super().__init__(
            name="XGrad-CAM",
            backbone=backbone,
            activation_weight="axiom",
        )
        return
