#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class FakeCAM(BaseCAM):
    """Fake-CAM

    Revisiting The Evaluation of Class Activation Mapping for Explainability:
    A Novel Metric and Experimental Analysis

    https://arxiv.org/abs/2104.10252

    Args:
        backbone (Backbone): resouce of CNN.
    """

    def __init__(
        self: FakeCAM,
        backbone: Backbone,
    ) -> None:
        super().__init__(
            name="Fake-CAM",
            backbone=backbone,
            position_weight="fake",
        )
        return
