#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Optional, Any

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class GroupCAM(BaseCAM):
    """Group-CAM

    "Group-CAM:
    Group Score-Weighted Visual Explanations
    for Deep Convolutional Networks"

    https://arxiv.org/abs/2103.13859

    Args:
        backbone (Backbone): resouce of CNN.
    """

    def __init__(
        self: GroupCAM,
        backbone: Backbone,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="Group-CAM",
            backbone=backbone,
            activation_weight="gradient",
            channel_weight="abscission",
            channel_group="k-means",
            random_state=random_state,
        )
        return
