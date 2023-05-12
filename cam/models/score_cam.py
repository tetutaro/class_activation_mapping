#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Any

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
    """

    def __init__(
        self: ScoreCAM,
        backbone: Backbone,
        n_channels: int = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="Score-CAM",
            backbone=backbone,
            channel_weight="abscission",
            n_channels=n_channels,
        )
        return
