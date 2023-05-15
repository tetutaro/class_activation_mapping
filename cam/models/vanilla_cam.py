#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Optional, Any

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class VanillaCAM(BaseCAM):
    """CAM

    B. Zhou, et al.
    "Learning Deep Features for Discriminative Localization"
    CVPR 2016.

    https://arxiv.org/abs/1512.04150

    Args:
        backbone (Backbone): resouce of CNN.
        random_state (Optional[int]): the random seed.

    Attributes:
        cam_name (str): the name of this CAM model.
    """

    cam_name: str = "CAM"

    def __init__(
        self: VanillaCAM,
        backbone: Backbone,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            backbone=backbone,
            activation_weight="class",
            random_state=random_state,
        )
