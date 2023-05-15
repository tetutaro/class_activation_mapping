#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Optional, Any

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class PolyCAM(BaseCAM):
    """Poly-CAM

    A. Englebert, et al.
    "Poly-CAM:
    High resolution class activation map for convolutional neural networks"
    ICPR 2022.

    https://arxiv.org/abs/2204.13359

    Args:
        backbone (Backbone): resouce of CNN.
        random_state (Optional[int]): the random seed.

    Attributes:
        cam_name (str): the name of this CAM model.
    """

    cam_name: str = "Poly-CAM"

    def __init__(
        self: PolyCAM,
        backbone: Backbone,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            backbone=backbone,
            activation_weight="gradient",
            gradient_gap=False,
            high_resolution=True,
            random_state=random_state,
        )
        return
