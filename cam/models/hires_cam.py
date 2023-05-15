#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Optional, Any

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class HiResCAM(BaseCAM):
    """HiRes-CAM

    R. Draelos, et al.
    "Use HiResCAM instead of Grad-CAM
    for faithful explanations of convolutional neural networks"
    arXiv 2020.

    https://arxiv.org/abs/2011.08891

    Args:
        backbone (Backbone): resouce of CNN.
        random_state (Optional[int]): the random seed.

    Attributes:
        cam_name (str): the name of this CAM model.
    """

    cam_name: str = "HiRes-CAM"

    def __init__(
        self: HiResCAM,
        backbone: Backbone,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            backbone=backbone,
            activation_weight="gradient",
            gradient_gap=False,
            random_state=random_state,
        )
        return
