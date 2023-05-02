#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

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
        self.name = "Grad-CAM"
        return

    def _create_weights(self: GradCAM) -> Weights:
        weights: Weights = Weights()
        for gradient, (_, k, _, _) in self.gradients:
            weights.append(
                gradient.view(1, k, -1).mean(dim=2).view(1, k, 1, 1)
            )
        weights.finalize()
        return weights
