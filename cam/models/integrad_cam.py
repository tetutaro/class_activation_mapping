#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Optional, Any

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
        random_state (Optional[int]): the random seed.

    Attributes:
        cam_name (str): the name of this CAM model.
    """

    cam_name: str = "InteGrad-CAM"

    def __init__(
        self: InteGradCAM,
        backbone: Backbone,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            backbone=backbone,
            activation_weight="gradient",
            gradient_smooth="integral",
            random_state=random_state,
        )
        return
