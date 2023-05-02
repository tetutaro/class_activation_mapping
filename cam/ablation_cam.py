#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Any

import torch
from torch import Tensor

from cam.base_cam import BaseCAM
from cam.libs_cam import ResourceCNN, Weights


class AblationCAM(BaseCAM):
    """Ablation-CAM

    "Ablation-CAM:
    Visual Explanations for Deep Convolutional Network
    via Gradient-free Localization"

    https://openaccess.thecvf.com/content_WACV_2020/html/Desai_Ablation-CAM_Visual_Explanations_for_Deep_Convolutional_Network_via_Gradient-free_Localization_WACV_2020_paper.html

    Args:
        resource (ResourceCNN): resouce of CNN model.
    """

    def __init__(
        self: AblationCAM,
        resource: ResourceCNN,
    ) -> None:
        super().__init__(
            resource=resource,
            target="last",
        )
        return

    def _set_name(self: AblationCAM) -> None:
        self.name_ = "Ablation-CAM"
        return

    def _create_weights(self: AblationCAM, **kwargs: Any) -> Weights:
        assert len(self.activations_) == 1
        activation, (_, k, _, _) = self.activations_[0]
        weight_list: List[float] = list()
        for i in range(k):
            # drop i-th channel (ablation)
            ablated_map: Tensor = activation.clone()
            ablated_map[:, i, :, :] = 0.0
            # forward classifier and calc reduced activation score
            with torch.no_grad():
                r_scores: Tensor = self._forward_classifier(ablated_map)
            reduced_score: float = (
                r_scores.detach().cpu().squeeze().numpy()[self.target_class_]
            )
            # calc "instantaneous slope" of score = weight of channel
            weight_list.append(
                (self.score_ - reduced_score) / (self.score_ + self.eps_)
            )
        weights: Weights = Weights()
        weights.append(torch.tensor(weight_list).view(1, k, 1, 1))
        weights.finalize()
        return weights
