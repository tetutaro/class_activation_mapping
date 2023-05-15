#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Optional, Any

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class AblationCAM(BaseCAM):
    """Ablation-CAM

    H. Ramaswamy, et al.
    "Ablation-CAM:
    Visual Explanations for Deep Convolutional Network
    via Gradient-free Localization"
    WACV 2020.

    https://openaccess.thecvf.com/content_WACV_2020/html/Desai_Ablation-CAM_Visual_Explanations_for_Deep_Convolutional_Network_via_Gradient-free_Localization_WACV_2020_paper.html

    Args:
        backbone (Backbone): resouce of CNN.
        random_state (Optional[int]): the random seed.

    Attributes:
        cam_name (str): the name of this CAM model.
    """

    cam_name: str = "Ablation-CAM"

    def __init__(
        self: AblationCAM,
        backbone: Backbone,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            backbone=backbone,
            channel_weight="ablation",
            random_state=random_state,
        )
        return
