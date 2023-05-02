#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

from cam.base_cam import BaseCAM
from cam.libs_cam import ResourceCNN, TargetLayer, Weights


class DummyCAM(BaseCAM):
    """Dummy-CAM

    add the dummy weight (all values = 1) to activation.

    Args:
        resource (ResourceCNN): resouce of CNN model.
        target (TargetLayer): target Conv. Layers to retrieve activations.
    """

    def __init__(
        self: DummyCAM,
        resource: ResourceCNN,
        target: TargetLayer = "last",
    ) -> None:
        super().__init__(
            resource=resource,
            target=target,
        )
        return

    def _set_name(self: DummyCAM) -> None:
        self.name = "Dummy-CAM"
        return

    def _create_weights(self: DummyCAM) -> Weights:
        return self._create_dummy_weights()
