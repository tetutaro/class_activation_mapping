#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Optional, Any

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class ScoreCAM(BaseCAM):
    """Score-CAM

    H. Wang, et al.
    "Score-CAM:
    Score-Weighted Visual Explanations for Convolutional Neural Networks"
    CVF 2020.

    https://arxiv.org/abs/1910.01279

    SIGCAM

    Q. Zhang, et al.
    "A Novel Visual Interpretability for Deep Neural Networks
    by Optimizing Activation Maps with Perturbation"
    AAAI 2021.

    Args:
        backbone (Backbone): resouce of CNN.
        n_channels (int): number of channels using for calc CIC score.
        random_state (Optional[int]): the random seed.

    Attributes:
        cam_name (str): the name of this CAM model.
    """

    cam_name: str = "Score-CAM"

    def __init__(
        self: ScoreCAM,
        backbone: Backbone,
        n_channels: int = -1,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            backbone=backbone,
            channel_weight="abscission",
            n_channels=n_channels,
            random_state=random_state,
        )
        return
