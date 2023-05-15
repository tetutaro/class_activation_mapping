#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Optional, Any

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class EigenCAM(BaseCAM):
    """Eigen-CAM

    M. Muhammad, et al.
    "Eigen-CAM:
    Class Activation Map using Principal Components"
    IJCNN 2020.

    https://arxiv.org/abs/2008.00299

    Args:
        backbone (Backbone): resouce of CNN.
        random_state (Optional[int]): the random seed.

    Attributes:
        cam_name (str): the name of this CAM model.
    """

    cam_name: str = "Eigen-CAM"

    def __init__(
        self: EigenCAM,
        backbone: Backbone,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            backbone=backbone,
            channel_weight="eigen",
            random_state=random_state,
        )
        return
