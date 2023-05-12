#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Any

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class InteGradCAM(BaseCAM):
    """InteGrad-CAM

    IntegratedGrad + Grad-CAM

    M. Sundararajan, et al.
    "Axiomatic Attribution for Deep Networks" (IntegratedGrad)
    ICML 2017.

    https://arxiv.org/abs/1703.01365

    Args:
        backbone (Backbone): resouce of CNN.
    """

    def __init__(
        self: InteGradCAM,
        backbone: Backbone,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="InteGrad-CAM",
            backbone=backbone,
            activation_weight="gradient",
            gradient_smooth="integral",
        )
        return
