#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Optional, Any

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class GradCAMpp(BaseCAM):
    """Grad-CAM++

    A. Chattopadhyay, et al.
    "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks"
    WACV 2018.

    https://arxiv.org/abs/1710.11063

    Args:
        backbone (Backbone): resouce of CNN.
        random_state (Optional[int]): the random seed.

    Attributes:
        cam_name (str): the name of this CAM model.
    """

    cam_name: str = "Grad-CAM++"

    def __init__(
        self: GradCAMpp,
        backbone: Backbone,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            backbone=backbone,
            activation_weight="gradient++",
            random_state=random_state,
        )
        return
