#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Any

from cam.base_cam import BaseCAM
from cam.libs_cam import ResourceCNN, Weights


class VanillaCAM(BaseCAM):
    """CAM

    "Learning Deep Features for Discriminative Localization"

    https://arxiv.org/abs/1512.04150

    Args:
        resource (ResourceCNN): resouce of CNN model.
        target (TargetLayer): target Conv. Layers to retrieve activations.
    """

    def __init__(
        self: VanillaCAM,
        resource: ResourceCNN,
    ) -> None:
        super().__init__(
            resource=resource,
            target="last",
        )

    def _set_name(self: VanillaCAM) -> None:
        self.name_ = "CAM"
        return

    def _create_weights(self: VanillaCAM, **kwargs: Any) -> Weights:
        return self._class_weights()
