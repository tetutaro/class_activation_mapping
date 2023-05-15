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
        n_channels (int): the number of abscission channel groups to calc.
        n_groups (Optional[int]): the number of channel groups.
        random_state (Optional[int]): the random seed.

    Attributes:
        cam_name (str): the name of this CAM model.
    """

    cam_name: str = "Group-CAM"

    def __init__(
        self: GroupCAM,
        backbone: Backbone,
        n_channels: int = -1,
        n_groups: Optional[int] = None,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            backbone=backbone,
            activation_weight="gradient",
            channel_weight="abscission",
            channel_group="k-means",
            n_channels=n_channels,
            n_groups=n_groups,
            random_state=random_state,
        )
        return
