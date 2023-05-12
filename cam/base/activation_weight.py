#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Callable, Any

import torch
from torch import Tensor

from cam.base import channel_shape, CommonWeight
from cam.base.containers import Weights, SaliencyMaps
from cam.base.context import Context


class ActivationWeight(CommonWeight):
    """A part of the CAM model that is responsible for raw saliency maps.

    ActivationWeight (output of Conv. Layer(s) when forwarding) represent
    which region of image was payed attention to recognize the target object.

    XXX
    """

    def __init__(self: ActivationWeight, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # store flags
        self.activation_weight: str = kwargs["activation_weight"]
        self.gradient_gap_: bool = kwargs["gradient_gap"]
        self.gradient_smooth: str = kwargs["gradient_smooth"]
        if self.activation_weight in ["none", "class"]:
            self.gradient_gap_ = False
        self.class_weights_: Tensor
        return

    def set_class_weights(
        self: ActivationWeight, class_weights: Tensor
    ) -> None:
        """set class weights

        Args:
            class_weights (Tensor): class weights
        """
        self.class_weights_ = class_weights
        return

    def _fake_smaps(self: ActivationWeight, ctx: Context) -> SaliencyMaps:
        """create fake saliency map whose values are almost 1.
        (except the top left corner)

        Args:
            ctx (Context): the context of this process.

        Returns:
            SaliencyMaps: fake saliency maps.
        """
        fake_smaps: SaliencyMaps = SaliencyMaps()
        for activation in ctx.activations:
            fake: Tensor = torch.ones_like(activation).to(self.device)
            fake[:, :, 0, 0] = 0.0  # set 0 to the top-left corner
            fake_smaps.append(fake)
        fake_smaps.finalize()
        return fake_smaps

    def _activation_weight_none(
        self: ActivationWeight,
        ctx: Context,
    ) -> Weights:
        """returns the dummy weight (each values of weight = 1).

        Args:
            ctx (Context): the context of this process.

        Returns:
            Weights: the dummy weight.
        """
        weights: Weights = Weights()
        for _ in range(len(ctx.activations)):
            weights.append(torch.ones((1, 1, 1, 1)).to(self.device))
        weights.finalize()
        return weights

    def _activation_weight_class(
        self: ActivationWeight,
        ctx: Context,
    ) -> Weights:
        """use the weight of the target label in the classifier part in CNN

        Args:
            ctx (Context): the context of this process.

        Returns:
            Weights: the weight.
        """
        weights: Weights = Weights()
        weights.append(
            self.class_weights_[[ctx.label], :].view(
                1, channel_shape(self.class_weights_), 1, 1
            )
        )
        weights.finalize()
        return weights

    def _activation_weight_gradient(
        self: ActivationWeight,
        ctx: Context,
    ) -> Weights:
        """use the gradient.

        Args:
            ctx (Context): the context of this process.

        Returns:
            Weights: the gradient.
        """
        weights: Weights = Weights()
        for gradient in ctx.gradients:
            weights.append(gradient.clone())
        weights.finalize()
        return weights

    def _activation_weight_gradient_pp(
        self: ActivationWeight,
        ctx: Context,
    ) -> Weights:
        """use smoothed gradient whose 2nd derivative becomes 0.

        Args:
            ctx (Context): the context of this process.

        Returns:
            Weights: smoothed gradient.
        """
        weights: Weights = Weights()
        for activation, gradient in zip(ctx.activations, ctx.gradients):
            k: int = channel_shape(activation)
            # alpha = gradient ** 2 /
            #         2 * (gradient ** 2) + SUM(avtivation * (gradient ** 3))
            # SUM: sum per channels over positions
            alpha_numer: Tensor = gradient.pow(2)
            alpha_denom: Tensor = 2 * alpha_numer
            alpha_denom += (
                (activation * gradient.pow(3))
                .view(1, k, -1)
                .sum(dim=2)
                .view(1, k, 1, 1)
            )
            alpha_denom = alpha_denom.clamp(min=1.0)  # for stability
            alpha: Tensor = alpha_numer / alpha_denom
            # weight = alpha * gradient
            weights.append(alpha * gradient)
        weights.finalize()
        return weights

    def _activation_weight_axiom(
        self: ActivationWeight,
        ctx: Context,
    ) -> Weights:
        """use gradient * activation.

        Args:
            ctx (Context): the context of this process.

        Returns:
            Weights: gradient * activation.
        """
        weights: Weights = Weights()
        for activation, gradient in zip(ctx.activations, ctx.gradients):
            weights.append(gradient * activation)
        weights.finalize()
        return weights

    # ## main function

    def weight_activation(
        self: ActivationWeight,
        ctx: Context,
    ) -> SaliencyMaps:
        """weight activation.
        Args:
            ctx (Context): the context of this process.

        Returns:
            SaliencyMaps: weighted activation = saliency map
        """
        # special case
        if self.activation_weight == "fake":
            return self._fake_smaps(ctx=ctx)
        # get weight for activation
        fn: Callable[[Context], Weights]
        if self.activation_weight == "none":
            fn = self._activation_weight_none
        elif self.activation_weight == "class":
            fn = self._activation_weight_class
        elif self.activation_weight == "gradient":
            fn = self._activation_weight_gradient
        elif self.activation_weight == "gradient++":
            fn = self._activation_weight_gradient_pp
        elif self.activation_weight == "axiom":
            fn = self._activation_weight_axiom
        else:
            raise NotImplementedError(
                f"activation weight is invalid: {self.activation_weight}"
            )
        source: Weights = fn(ctx=ctx)
        # GAP (global average pooling) weight
        weights: Weights
        if self.gradient_gap_:
            weights = Weights()
            for weight in source:
                k: int = channel_shape(weight)
                weights.append(
                    weight.view(1, k, -1).mean(dim=2).view(1, k, 1, 1)
                )
            weights.finalize()
            source.clear()
        else:
            # do nothing on purpose
            weights = source
        # create saliency maps by multiplying weight and activation
        raw_smaps: SaliencyMaps = SaliencyMaps()
        for weight, activation in zip(weights, ctx.activations):
            raw_smaps.append(smap=weight * activation)
        raw_smaps.finalize()
        weights.clear()
        return raw_smaps
