#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Optional, Any

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
        target (TargetLayer): target Conv. Layers to retrieve activations.
    """

    def __init__(
        self: SmoothGradCAMpp,
        resource: ResourceCNN,
        target: TargetLayer = "last",
    ) -> None:
        super().__init__(
            resource=resource,
            target=target,
        )
        return

    def _set_name(self: SmoothGradCAMpp) -> None:
        self.name_ = "Smooth Grad-CAM++"
        return

    def _create_weights(
        self: SmoothGradCAMpp,
        random_state: Optional[int],
        n_samples: Optional[int] = 5,
        sigma: Optional[float] = 0.3,
        **kwargs: Any,
    ) -> Weights:
        """
        Args:
            random_state (int): random seed
            n_samples (int): number of samplings
            sigma (float): standard deviation of noise
        """
        if random_state is not None:
            # set random seeds
            np.random.seed(seed=random_state)
            torch.manual_seed(seed=random_state)
            torch.cuda.manual_seed(seed=random_state)
        # create dummy (value = 0) weights
        weight_list: List[Tensor] = list()
        for _, (_, k, _, _) in self.activations_:
            weight_list.append(torch.zeros((1, k, 1, 1)).to(self.device_))
        # clac weights using noised image and sum them up
        for _ in range(n_samples):
            # create noised image
            noised_image: Tensor = torch.normal(
                mean=self.image_, std=sigma
            ).to(self.device_)
            noised_image.requires_grad = True
            # forward network
            self._forward(image=noised_image, requires_grad=True)
            # calc weights of noised image
            noised_weights: Weights = self._grad_pp_weights()
            # sum them to dummy weights
            for i, (weight, _) in enumerate(noised_weights):
                weight_list[i] += weight
        weights: Weights = Weights()
        for weight in weight_list:
            weights.append(weight)
        weights.finalize()
        return weights
