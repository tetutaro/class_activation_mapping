#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Any

from cam.base_cam import BaseCAM
from cam.libs_cam import ResourceCNN, TargetLayer, Weights


class ScoreCAM(BaseCAM):
    """Score-CAM

    "Score-CAM:
    Score-Weighted Visual Explanations for Convolutional Neural Networks"

    https://arxiv.org/abs/1910.01279

    Args:
        resource (ResourceCNN): resouce of CNN model.
        target (TargetLayer): target Conv. Layers to retrieve activations.
        n_channels (int):
            number of channels using for calc saliency map.
            if -1, use all channels.
    """

    def __init__(
        self: ScoreCAM,
        resource: ResourceCNN,
        target: TargetLayer = "last",
    ) -> None:
        super().__init__(
            resource=resource,
            target=target,
            channel_weight="score",
        )
        return

    def _set_name(self: ScoreCAM) -> None:
        self.name_ = "Score-CAM"
        return

    def _create_weights(self: ScoreCAM, **kwargs: Any) -> Weights:
        return self._dummy_weights()
