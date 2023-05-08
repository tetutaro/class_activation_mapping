#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

from cam.base.base_cam import BaseCAM
from cam.backbones.backbone import Backbone


class DummyCAM(BaseCAM):
    """Dummy-CAM

    add the dummy weight (all values = 1) to activation.

    Args:
        backbone (Backbone): resouce of CNN.
    """

    def __init__(
        self: DummyCAM,
        backbone: Backbone,
    ) -> None:
        super().__init__(
            name="Dummy-CAM",
            backbone=backbone,
        )
        return
