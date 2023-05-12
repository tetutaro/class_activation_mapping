#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Any

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class FakeCAM(BaseCAM):
    """Fake-CAM

    S. Poppi, et al.
    "Revisiting The Evaluation of Class Activation Mapping for Explainability:
    A Novel Metric and Experimental Analysis"
    CVPR 2021.

    https://arxiv.org/abs/2104.10252

    Args:
        backbone (Backbone): resouce of CNN.
    """

    def __init__(
        self: FakeCAM,
        backbone: Backbone,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="Fake-CAM",
            backbone=backbone,
            activation_weight="fake",
        )
        return
