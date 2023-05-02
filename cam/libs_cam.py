#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""libraries for CAM models"""
from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Callable, Union, TypedDict

from torch import Tensor
from torchvision.models._api import WeightsEnum

# the shape for image, activation, gradient, weigt, ...
Shape = Tuple[int, int, int, int]
# the type of each cached items
CacheItem = Tuple[Tensor, Shape]
# the function to get the size of batch
batch_size: Callable = lambda x: x.size()[0]
# the function to get the size of channel
channel_size: Callable = lambda x: x.size()[1]
# how to indicate the target layers
TargetLayer = Optional[Union[int, List[int], str]]


class ResourceCNN(TypedDict):
    """the resource container class of CNN model.

    Args:
        net (Callable): the function creates the CNN module.
        weights (WeightsEnum): the pre-trained weights of the CNN model.
    """

    net: Callable
    weights: WeightsEnum


class Activations:
    """the cache of activations (output of Conv. Layers when forwarding)."""

    def __init__(self: Activations) -> None:
        self.dict_cache_: Dict[Tuple[int, int], Tensor] = dict()
        self.list_cache_: Optional[List[CacheItem]] = None
        return

    def append(self: Activations, output: Tensor) -> None:
        """append an activation to the cache.

        if another activaion of the same shape is already stored, overwrite it
        (store only the last activation which has the same shape).

        Args:
            output (Tensor): an activation (output of Conv. Layer) to append.
        """
        activation: Tensor = output.clone().detach()
        assert batch_size(activation) == 1
        shape: Shape = activation.size()
        self.dict_cache_[shape] = activation
        return

    def finalize(self: Activations) -> None:
        """sort the cache and set them in the list in shape order."""
        self.list_cache_ = [
            (x[1], x[0])
            for x in sorted(
                self.dict_cache_.items(),
                key=lambda y: (-y[0][1], y[0][2], y[0][3]),
            )
        ]
        return

    def clear(self: Activations) -> None:
        """clear the cache"""
        self.dict_cache_ = dict()
        self.list_cache_ = None
        return

    def __getitem__(self: Activations, idx: int) -> CacheItem:
        """returns the indicated activation.

        Args:
            idx (int): the offset of cached activation.

        Returns:
            Tensor: the indicated activation.
        """
        if self.list_cache_ is None:
            raise SystemError("forget finalizing")
        if idx >= len(self.list_cache_):
            raise IndexError()
        return self.list_cache_[idx]

    def __len__(self: Activations) -> int:
        """returns the number of cached activations.

        Returns:
            int: the number of cached activations.
        """
        if self.list_cache_ is None:
            raise SystemError("forget finalizing")
        return len(self.list_cache_)

    def __str__(self: Activations) -> str:
        """returns shapes of activations.

        Returns:
            str: shapes of activations
        """
        if self.list_cache_ is None:
            raise SystemError("forget finalizing")
        return ", ".join(
            ["x".join([str(s) for s in x[1]]) for x in self.list_cache_]
        )


class Gradients(Activations):
    """the cache of gradients (output of Conv. Layers when backwarding)."""

    def append(self: Gradients, output: Tensor) -> None:
        """append a gradient to the cache.

        if another gradient of the same shape is already stored, ignore output
        (store only the last gradient which has the same shape).

        Args:
            output (Tensor): a gradient (output of Conv. Layer) to append.
        """
        gradient: Tensor = output.clone().detach()
        assert batch_size(gradient) == 1
        shape: Shape = gradient.size()
        if self.dict_cache_.get(shape) is None:
            self.dict_cache_[shape] = gradient
        return


class Weights(Activations):
    """the cache of weights for activations"""

    def __init__(self: Weights) -> None:
        self.list_cache_: List[CacheItem] = list()
        return

    def append(self: Weights, weight: Tensor) -> None:
        """append the weight to the cache.

        Args:
            weight (Tensor): a weight to append.
        """
        assert batch_size(weight) == 1
        shape: Shape = weight.size()
        self.list_cache_.append((weight, shape))
        return

    def finalize(self: Weights) -> None:
        """dummy function"""
        return

    def clear(self: Weights) -> None:
        """clear the cache"""
        self.list_cache_ = list()
        return


class SaliencyMaps(Weights):
    """the cache of saliency maps"""

    def __init__(
        self: SaliencyMaps,
        is_layer: bool = False,
    ) -> None:
        super().__init__()
        self.is_layer_ = is_layer
        return

    def append(self: SaliencyMaps, smap: Tensor) -> None:
        """append the saliency map to the cache.

        Args:
            smap (Tensor): a saliency map to append.
        """
        if self.is_layer_:
            assert channel_size(smap) == 1
        super().append(weight=smap)
        return
