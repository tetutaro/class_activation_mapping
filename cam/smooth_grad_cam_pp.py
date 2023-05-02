#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List

import numpy as np
import torch
from torch import Tensor

from cam.base_cam import BaseCAM
from cam.libs_cam import ResourceCNN, TargetLayer, Weights


class SmoothGradCAMpp(BaseCAM):
    """Smooth Grad-CAM++

    "Smooth Grad-CAM++:
    An Enhanced Inference Level Visualization Technique
    for Deep Convolutional Neural Network Models"

    https://arxiv.org/abs/1908.01224

    Args:
        resource (ResourceCNN): resouce of CNN model.
        n_samples (int): number of samplings
        sigma (float): standard deviation of noise
        random_state (int): random seed
        target (TargetLayer): target Conv. Layers to retrieve activations.
    """

    def __init__(
        self: SmoothGradCAMpp,
        resource: ResourceCNN,
        n_samples: int,
        sigma: float,
        random_state: int,
        target: TargetLayer = "last",
    ) -> None:
        super().__init__(
            resource=resource,
            target=target,
        )
        self.n_samples: int = n_samples
        self.sigma: float = sigma
        # set seeds of random
        np.random.seed(seed=random_state)
        torch.manual_seed(seed=random_state)
        torch.cuda.manual_seed(seed=random_state)
        return

    def _set_name(self: SmoothGradCAMpp) -> None:
        self.name = "Smooth Grad-CAM++"
        return

    def _create_weights(self: SmoothGradCAMpp) -> Weights:
        # append dummy (value = 0) weights
        weight_list: List[Tensor] = list()
        for _, (_, k, _, _) in self.activations:
            weight_list.append(torch.zeros((1, k, 1, 1)).to(self.device))
        # clac weights using noised image and sum them up
        for _ in range(self.n_samples):
            # create noised image
            noised_image: Tensor = torch.normal(
                mean=self.image, std=self.sigma
            ).to(self.device)
            noised_image.requires_grad = True
            # forward network
            self._forward(image=noised_image, requires_grad=True)
            for i, ((activation, (_, k, _, _)), (gradient, _)) in enumerate(
                zip(self.activations, self.gradients)
            ):
                # calc alpha (like Grad-CAM++)
                alpha_numer: Tensor = gradient.pow(2.0)
                alpha_denom: Tensor = 2.0 * alpha_numer.clone()
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
                # omit ReLU to visualize negative regions
                weight_list[i] += (
                    (np.exp(self.score) * alpha * gradient)
                    .view(1, k, -1)
                    .sum(dim=2)
                    .view(1, k, 1, 1)
                )
        weights: Weights = Weights()
        for weight in weight_list:
            weights.append(weight)
        weights.finalize()
        return weights
