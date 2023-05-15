#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Optional, Any

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class GradCAM(BaseCAM):
    """Grad-CAM

    R. Selvaraju, et al.
    "Grad-CAM:
    Visual Explanations from Deep Networks via Gradient-based Localization"
    ICCV 2017.

    https://arxiv.org/abs/1610.02391

    Args:
        backbone (Backbone): resouce of CNN.
        random_state (Optional[int]): the random seed.

    Attributes:
        cam_name (str): the name of this CAM model.
    """

    cam_name: str = "Grad-CAM"

    def __init__(
        self: GradCAM,
        backbone: Backbone,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            backbone=backbone,
            activation_weight="gradient",
            random_state=random_state,
        )
        return
