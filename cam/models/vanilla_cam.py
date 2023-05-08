#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class VanillaCAM(BaseCAM):
    """CAM

    "Learning Deep Features for Discriminative Localization"

    https://arxiv.org/abs/1512.04150

    Args:
        backbone (Backbone): resouce of CNN.
    """

    def __init__(
        self: VanillaCAM,
        backbone: Backbone,
    ) -> None:
        super().__init__(
            name="CAM",
            backbone=backbone,
            activation_weight="class",
        )
