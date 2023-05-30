#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""the module contains the class "Acquired" and functions about that."""
from __future__ import annotations
from typing import List, NamedTuple

import torch
from torch import Tensor

from cam.base import DEBUG, Shape, batch_shape, channel_shape
from cam.base.containers import Weights


class Acquired(NamedTuple):
    """Activations and Gradients
    that is the result of forwarding the image to CNN."""

    activations: Weights
    gradients: Weights
    scores: Tensor

    def finalize(self: Acquired) -> None:
        """debug function."""
        if DEBUG:
            n_layers_a: int = len(self.activations)
            n_layers_g: int = len(self.gradients)
            assert n_layers_a > 0 and n_layers_a == n_layers_g
            n_batches: int = batch_shape(self.activations[0])
            for activation, gradient in zip(self.activations, self.gradients):
                assert activation.size() == gradient.size()
                assert batch_shape(activation) == n_batches
            assert batch_shape(self.scores) == n_batches
        return

    def clear(self: Acquired) -> None:
        """clear this instance."""
        self.activations.clear()
        self.gradients.clear()
        return


def merge_acquired_list(acquired_list: List[Acquired]) -> Acquired:
    """merge acquireds.

    Args:
        acquired_list (List[Forwarded]): list of acquireds to be merged.

    Returns:
        Acquired: merged Acquired.
    """
    n_batches: int = len(acquired_list)
    assert n_batches > 0
    if n_batches == 1:
        return acquired_list[0]
    # number of layers
    n_layers: int = len(acquired_list[0].activations)
    if DEBUG:
        shape_list: List[Shape] = [
            x.size() for x in acquired_list[0].activations
        ]
    # distibute activation and gradient for each layer
    activation_lists: List[List[Weights]] = [[] for _ in range(n_layers)]
    gradient_lists: List[List[Weights]] = [[] for _ in range(n_layers)]
    scores_list: List[Tensor] = list()
    for acquired in acquired_list:
        for layer, (activation, gradient) in enumerate(
            zip(acquired.activations, acquired.gradients)
        ):
            if DEBUG:
                assert activation.size() == shape_list[layer]
                assert gradient.size() == shape_list[layer]
            activation_lists[layer].append(activation)
            gradient_lists[layer].append(gradient)
        scores_list.append(acquired.scores)
    if DEBUG:
        for i, alist in enumerate(activation_lists):
            assert len(alist) == n_batches
        for i, glist in enumerate(gradient_lists):
            assert len(glist) == n_batches
        len(scores_list) == n_batches
    # merge activations
    activations: Weights = Weights()
    for activaion_list in activation_lists:
        activations.append(weight=torch.cat(activaion_list, dim=0))
    activations.finalize()
    # merge gradients
    gradients: Weights = Weights()
    for gradient_list in gradient_lists:
        gradients.append(weight=torch.cat(gradient_list, dim=0))
    gradients.finalize()
    # merge scores
    scores: Tensor = torch.cat(scores_list, dim=0)
    # create fowarded
    merged: Acquired = Acquired(
        activations=activations,
        gradients=gradients,
        scores=scores,
    )
    merged.finalize()
    # clear original forwardeds
    while True:
        try:
            acquired: Acquired = acquired_list.pop()
            acquired.clear()
        except IndexError:
            break
    return merged


def merge_acquired(
    acquired: Acquired,
    unit_size: int,
    merge_type: str,
) -> Acquired:
    """merge acquired over units.

    Args:
        acquired (Acquired): the acquired to be merged.
        unit_size (int): unit size.

    Returns:
        Acquired: integrated acquired.
    """
    activations: Weights = Weights()
    gradients: Weights = Weights()
    # merge for each layer
    for activation, gradient in zip(acquired.activations, acquired.gradients):
        shape: Shape = activation.size()
        if DEBUG:
            assert shape[0] % unit_size == 0
            assert gradient.size() == shape
        # merge activation
        if merge_type == "smooth":
            activations.append(
                activation.view(-1, unit_size, *shape[1:]).mean(dim=1)
            )
        else:  # merge_type == "integral"
            activations.append(
                activation.view(-1, unit_size, *shape[1:])[:, -1, ...]
            )
        # merge gradient
        gradients.append(gradient.view(-1, unit_size, *shape[1:]).mean(dim=1))
    activations.finalize()
    gradients.finalize()
    # merge scores
    n_labels: int = channel_shape(acquired.scores)
    scores: Tensor
    if merge_type == "smooth":
        scores = acquired.scores.view(-1, unit_size, n_labels).mean(dim=1)
    else:  # merge_type == "integral"
        scores = acquired.scores.view(-1, unit_size, n_labels)[:, -1, :]
    # create merged acquired
    merged: Acquired = Acquired(
        activations=activations,
        gradients=gradients,
        scores=scores,
    )
    merged.finalize()
    # clear original acquired
    acquired.clear()
    return merged
