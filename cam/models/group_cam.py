#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Optional, Any

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class GroupCAM(BaseCAM):
    """Group-CAM

    Q. Zhang, et al.
    "Group-CAM:
    Group Score-Weighted Visual Explanations
    for Deep Convolutional Networks"
    arXiv 2021.

    https://arxiv.org/abs/2103.13859

    Args:
        backbone (Backbone): resouce of CNN.
    """

    def __init__(
        self: GroupCAM,
        backbone: Backbone,
        n_channels: int = -1,
        n_channel_groups: Optional[int] = None,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="Group-CAM",
            backbone=backbone,
            activation_weight="gradient",
            channel_weight="abscission",
            channel_group="k-means",
            n_channels=n_channels,
            n_channel_groups=n_channel_groups,
            random_state=random_state,
        )
        return
