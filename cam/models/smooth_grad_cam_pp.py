#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Optional, Any

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class SmoothGradCAMpp(BaseCAM):
    """Smooth Grad-CAM++

    "Smooth Grad-CAM++:
    An Enhanced Inference Level Visualization Technique
    for Deep Convolutional Neural Network Models"

    https://arxiv.org/abs/1908.01224

    Args:
        backbone (Backbone): resouce of CNN.
        n_samples (int): number of samplings. (use it in SmoothGrad)
        sigma (float): sdev of Normal Dist. (use it in SmoothGrad)
        random_state (Optional[int]): the random seed.
    """

    def __init__(
        self: SmoothGradCAMpp,
        backbone: Backbone,
        n_samples: int = 8,
        sigma: float = 0.3,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="Smooth Grad-CAM++",
            backbone=backbone,
            activation_weight="gradient++",
            gradient_smooth="noise",
            n_samples=n_samples,
            sigma=sigma,
            random_state=random_state,
        )
        return
