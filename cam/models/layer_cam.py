#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Any

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class LayerCAM(BaseCAM):
    """Layer-CAM

    "LayerCAM:
    Exploring Hierarchical Class Activation Maps for Localization"

    https://mmcheng.net/mftp/Papers/21TIP_LayerCAM.pdf

    Args:
        backbone (Backbone): resouce of CNN.
    """

    def __init__(
        self: LayerCAM,
        backbone: Backbone,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="Layer-CAM",
            backbone=backbone,
            activation_weight="gradient",
            gradient_gap=False,
        )
        return
