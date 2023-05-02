#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Any

from cam.base_cam import BaseCAM
from cam.libs_cam import ResourceCNN, TargetLayer, Weights


class GradCAMpp(BaseCAM):
    """Grad-CAM++

    "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks"

    https://arxiv.org/abs/1710.11063

    Args:
        resource (ResourceCNN): resouce of CNN model.
        target (TargetLayer): target Conv. Layers to retrieve activations.
    """

    def __init__(
        self: GradCAMpp,
        resource: ResourceCNN,
        target: TargetLayer = "last",
    ) -> None:
        super().__init__(
            resource=resource,
            target=target,
            requires_grad=True,
        )
        return

    def _set_name(self: GradCAMpp) -> None:
        self.name_ = "Grad-CAM++"
        return

    def _create_weights(self: GradCAMpp, **kwargs: Any) -> Weights:
        return self._grad_pp_weights()
