#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Any

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class AblationCAM(BaseCAM):
    """Ablation-CAM

    "Ablation-CAM:
    Visual Explanations for Deep Convolutional Network
    via Gradient-free Localization"

    https://openaccess.thecvf.com/content_WACV_2020/html/Desai_Ablation-CAM_Visual_Explanations_for_Deep_Convolutional_Network_via_Gradient-free_Localization_WACV_2020_paper.html

    Args:
        backbone (Backbone): resouce of CNN.
    """

    def __init__(
        self: AblationCAM,
        backbone: Backbone,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="Ablation-CAM",
            backbone=backbone,
            channel_weight="ablation",
        )
        return
