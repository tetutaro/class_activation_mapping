#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Optional, Any

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class CosineCAM(BaseCAM):
    """Cosine-CAM

    normalize channel-position matrix before k-Means.

    Args:
        backbone (Backbone): resouce of CNN.
        n_channels (int): the number of abscission channel groups to calc.
        n_groups (Optional[int]): the number of channel groups.
        random_state (Optional[int]): the random seed.

    Attributes:
        cam_name (str): the name of this CAM model.
    """

    cam_name: str = "Cosine-CAM"

    def __init__(
        self: CosineCAM,
        backbone: Backbone,
        n_channels: int = -1,
        n_groups: Optional[int] = None,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            backbone=backbone,
            activation_weight="axiom",
            channel_weight="abscission",
            channel_group="k-means",
            channel_cosine=True,
            n_channels=n_channels,
            n_groups=n_groups,
            random_state=random_state,
        )
        return
