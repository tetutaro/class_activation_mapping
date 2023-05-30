#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""the module of containers for activations and gradients

Each activation and gradient have
different number of width, height and channels by the layer they was retrieved.
(But each number of batch is the same)
"""
from __future__ import annotations
from typing import List, Dict

from torch import Tensor

from cam.base import DEBUG, Shape, batch_shape


class Weights:
    """the contaner of weights"""

    def __init__(self: Weights) -> None:
        self.list_cache_: List[Tensor] = list()
        return

    def append(self: Weights, weight: Tensor) -> None:
        """append the weight to the container.

        Args:
            weight (Tensor): a weight to append.
        """
        self.list_cache_.append(weight)
        return

    def finalize(self: Weights) -> None:
        """debug function."""
        if DEBUG:
            if len(self.list_cache_) > 0:
                b: int = batch_shape(self.list_cache_[0])
                for weight in self.list_cache_:
                    assert batch_shape(weight) == b
        return

    def clone(self: Weights) -> Weights:
        """clone self.

        Returns:
            Weight: cloned weights.
        """
        cloned: Weights = Weights()
        for weight in self.list_cache_:
            cloned.append(weight.clone())
        cloned.finalize()
        return cloned

    def clear(self: Weights) -> None:
        """clear the container."""
        while True:
            try:
                weight: Tensor = self.list_cache_.pop()  # noqa
                del weight
            except IndexError:
                break
        self.list_cache_ = list()
        return

    def __getitem__(self: Weights, idx: int) -> Weights:
        """returns the indicated activation.

        Args:
            idx (int): the offset of cached activation.

        Returns:
            Tensor: the indicated activation.
        """
        if idx >= len(self.list_cache_):
            raise IndexError()
        return self.list_cache_[idx]

    def __len__(self: Weights) -> int:
        """returns the number of cached weights.

        Returns:
            int: the number of cached weights.
        """
        if self.list_cache_ is None:
            raise SystemError("forget finalizing")
        return len(self.list_cache_)

    def __str__(self: Weights) -> str:
        """returns shapes of each weight in the cache.

        Returns:
            str: shapes of each weight in the cache.
        """
        if self.list_cache_ is None:
            raise SystemError("forget finalizing")
        return ", ".join(
            ["x".join([str(s) for s in x.size()]) for x in self.list_cache_]
        )


class SaliencyMaps(Weights):
    """the cache of saliency maps"""

    def append(self: SaliencyMaps, smap: Tensor) -> None:
        """append the saliency map to the container.

        Args:
            smap (Tensor): a saliency map to append.
        """
        super().append(weight=smap)
        return


class Activations(Weights):
    """the cache of activations (output of Conv. Layers when forwarding)."""

    def __init__(self: Activations) -> None:
        super().__init__()
        self.dict_cache_: Dict[Shape, Tensor] = dict()
        return

    def append(self: Activations, activation: Tensor) -> None:
        """append an activation to the container.

        if another activaion of the same shape is already stored, overwrite it
        (store only the last activation which has the same shape).

        Args:
            activation (Tensor): an activation to append.
        """
        shape: Shape = activation.size()
        self.dict_cache_[shape] = activation
        return

    def finalize(self: Activations) -> None:
        """sort the container and set them in the list in shape order."""
        self.list_cache_ = [
            x[1]
            for x in sorted(
                self.dict_cache_.items(),
                key=lambda y: (-y[0][1], y[0][2], y[0][3]),
            )
        ]
        super().finalize()
        return

    def clear(self: Activations) -> None:
        """clear the container."""
        self.dict_cache_ = dict()
        super().clear()
        return


class Gradients(Activations):
    """the cache of gradients (output of Conv. Layers when backwarding)."""

    def append(self: Gradients, gradient: Tensor) -> None:
        """append a gradient to the cache.

        if another gradient of the same shape is already stored, ignore output
        (store only the last gradient which has the same shape).

        Args:
            output (Tensor): a gradient (output of Conv. Layer) to append.
        """
        shape: Shape = gradient.size()
        if self.dict_cache_.get(shape) is None:
            self.dict_cache_[shape] = gradient
        return
