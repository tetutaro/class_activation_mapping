#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Any

from cam.base_cam import BaseCAM
from cam.libs_cam import ResourceCNN, TargetLayer, Weights


class GradCAM(BaseCAM):
    """Grad-CAM

    "Grad-CAM:
    Visual Explanations from Deep Networks via Gradient-based Localization"

    https://arxiv.org/abs/1610.02391

    Args:
        resource (ResourceCNN): resouce of CNN model.
        target (TargetLayer): target Conv. Layers to retrieve activations.
    """

    def __init__(
        self: GradCAM,
        resource: ResourceCNN,
        target: TargetLayer = "last",
    ) -> None:
        super().__init__(
            resource=resource,
            target=target,
            requires_grad=True,
        )
        return

    def _set_name(self: GradCAM) -> None:
        self.name_ = "Grad-CAM"
        return

    def _create_weights(self: GradCAM, **kwargs: Any) -> Weights:
        return self._grad_weights()
