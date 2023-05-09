#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Any

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class EigenCAM(BaseCAM):
    """Eigen-CAM

    "Eigen-CAM:
    Class Activation Map using Principal Components"

    https://arxiv.org/abs/2008.00299

    Args:
        backbone (Backbone): resouce of CNN.
    """

    def __init__(
        self: EigenCAM,
        backbone: Backbone,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="Eigen-CAM",
            backbone=backbone,
            channel_weight="eigen",
        )
        return
