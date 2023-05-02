#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

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
        self.name = "Grad-CAM++"
        return

    def _create_weights(self: GradCAMpp) -> Weights:
        weights: Weights = Weights()
        for (activation, (_, k, _, _)), (gradient, _) in zip(
            self.activations, self.gradients
        ):
            # calc alpha (the eq (19) in the paper)
            alpha_numer: Tensor = gradient.pow(2.0)
            alpha_denom: Tensor = 2.0 * alpha_numer
            alpha_denom += (
                (gradient.pow(3.0) * activation)
                .view(1, k, -1)
                .sum(dim=2)
                .view(1, k, 1, 1)
            )
            alpha_denom = (
                torch.where(
                    alpha_denom != 0.0,
                    alpha_denom,
                    torch.ones_like(alpha_denom),
                )
                + self.eps
            )  # for stability
            alpha: Tensor = alpha_numer / alpha_denom
            weights.append(
                (np.exp(self.score) * alpha * gradient)
                .view(1, k, -1)
                .sum(dim=2)
                .view(1, k, 1, 1)
            )
        weights.finalize()
        return weights
