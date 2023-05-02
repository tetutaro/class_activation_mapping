#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

from cam.base_cam import BaseCAM
from cam.libs_cam import ResourceCNN, TargetLayer, Weights


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
        target: TargetLayer = "last",
    ) -> None:
        self._assert_target_is_last(target=target)
        super().__init__(
            resource=resource,
            target=target,
        )

    def _set_name(self: VanillaCAM) -> None:
        self.name = "CAM"
        return

    def _create_weights(self: VanillaCAM) -> Weights:
        return self._extract_class_weights()
