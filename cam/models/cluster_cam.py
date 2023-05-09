#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Optional, Any

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class ClusterCAM(BaseCAM):
    """Cluster-CAM

    "Cluster-CAM:
    Cluster-Weighted Visual Interpretation of CNNs' Decision
    in Image Classification"

    https://arxiv.org/abs/2302.01642

    Args:
        backbone (Backbone): resouce of CNN.
    """

    def __init__(
        self: ClusterCAM,
        backbone: Backbone,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="Cluster-CAM",
            backbone=backbone,
            activation_weight="gradient",
            channel_weight="abscission",
            channel_group="k-means",
            channel_minmax=True,
            random_state=random_state,
        )
        return
