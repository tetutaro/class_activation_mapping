#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""common types, values, functions and CommonCAM"""
from __future__ import annotations
from typing import Tuple, List, Callable, Union, Any

import torch
from torch import Tensor

# debug flag
DEBUG: bool = True

# target names (used when target is indicated as string)
target_names: List[str] = [
    "all",
    "last",
]
# methods of smoothing activations and gradients
smooth_types: List[str] = [
    "none",  # do nothing
    "noise",  # SmoothGrad
    "integral",  # IntegratedGrads
    "noise+integral",  # SmoothGrad + IntegratedGrads
]
# types of weights for each activations
activation_weights: List[str] = [
    "none",  # all values == 1
    "class",  # CAM
    "gradient",  # Grad-CAM
    "gradient++",  # Grad-CAM++
    "axiom",  # XGrad-CAM
]
# methods of weighting for each positions
position_weights: List[str] = [
    "none",  # do nothing
    "fake",  # Fake-CAM
    "eigen",  # first eigen-vector (Eigen-CAM)
    "ablation",  # position-wise Ablation (Ablation-CAM)
    "abscission",  # position-wise Abscission (Score-CAM)
]
# methods of weighting for each channels
channel_weights: List[str] = [
    "none",  # average over channels
    "eigen",  # first eigen-vector (Eigen-CAM)
    "ablation",  # channel-wise Ablation (Ablation-CAM)
    "abscission",  # channel-wise Abscission (Score-CAM)
]
# clustering methods for positions/channels
group_types: List[str] = [
    "none",  # calc weights for each positions/channels
    "k-means",  # k-Means (Group-CAM)
    "spectral",  # Spectral Clustering (Cluster-CAM)
]
# method of merging layers
merge_layers: List[str] = [
    "none",  # sum over layers (Layer-CAM)
    "multiply",  # Poly-CAM
]

# original types
# type of target
TargetLayer = Union[int, List[int], str]
# type of shape for Weights, Saliency Maps, Activations, Gradients
Shape = Tuple[int, int, int, int]

# functions
# the function to get the size of batch
batch_shape: Callable[[Tensor], int] = lambda x: x.size()[0]
# the function to get the size of channel
channel_shape: Callable[[Tensor], int] = lambda x: x.size()[1]
# the function to get the size of channel
position_shape: Callable[[Tensor], Tuple[int, int]] = lambda x: tuple(
    x.size()[2:4]
)


class CommonSMAP:
    """the common class for
    Network, RawSMAPS, PositionSMAPS, ChannelSMAPS and FinalSMAP.
    """

    def __init__(self: CommonSMAP, **kargs: Any) -> None:
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.eps: float = 1e-6
        return
