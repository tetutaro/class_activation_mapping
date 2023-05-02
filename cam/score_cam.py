#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

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
        n_channels: int = -1,
    ) -> None:
        self._assert_target_is_last(target=target)
        super().__init__(
            resource=resource,
            target=target,
            requires_grad=False,
            channel_weight="score",
            n_channels=n_channels,
        )
        self.n_channels: int = n_channels
        return

    def _set_name(self: ScoreCAM) -> None:
        self.name = "Score-CAM"
        return

    def _create_weights(self: ScoreCAM) -> Weights:
        return self._extract_class_weights()
