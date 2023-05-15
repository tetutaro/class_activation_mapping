#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Optional, Any

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class XGradCAM(BaseCAM):
    """XGrad-CAM

    R. Fu, et al.
    "Axiom-based Grad-CAM:
    Towards Accurate Visualization and Explanation of CNNs"
    BMVC 2020.

    https://arxiv.org/abs/2008.02312

    Args:
        backbone (Backbone): resouce of CNN.
        random_state (Optional[int]): the random seed.

    Attributes:
        cam_name (str): the name of this CAM model.
    """

    cam_name: str = "XGrad-CAM"

    def __init__(
        self: XGradCAM,
        backbone: Backbone,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            backbone=backbone,
            activation_weight="axiom",
            random_state=random_state,
        )
        return
