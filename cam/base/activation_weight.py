#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Callable, Any

import torch
from torch import Tensor

from cam.base import channel_shape, position_shape, CommonWeight
from cam.base.containers import Weights, SaliencyMaps
from cam.base.context import Context


class ActivationWeight(CommonWeight):
    """A part of the CAM model that is responsible for raw saliency maps.

    ActivationWeight (output of Conv. Layer(s) when forwarding) represent
    which region of image was payed attention to recognize the target object.

    XXX

    Args:
        activation_weight (str): the type of weight for each activations.
        gradient_smooth (str): the method of smoothing gradient.
        gradient_no_gap (bool): if True, use gradient as is.
    """

    def __init__(
        self: ActivationWeight,
        activation_weight: str,
        gradient_smooth: str,
        gradient_no_gap: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # store flags
        self.activation_weight: str = activation_weight
        self.gradient_smooth: str = gradient_smooth
        self.gradient_no_gap_: bool = gradient_no_gap
        if self.activation_weight in ["none", "class"]:
            self.gradient_no_gap_ = True
        self.class_weight_: Tensor
        return

    def set_class_weight(
        self: ActivationWeight,
        class_weight: Tensor,
    ) -> None:
        """set class weights.

        Args:
            class_weight (Tensor): class weights.
        """
        self.class_weight_ = class_weight
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
            u, v = position_shape(activation)
            fake: Tensor = torch.ones(1, 1, u, v).to(self.device)
            fake[:, :, 0, 0] = 0.0  # set 0 to the top-left corner
            fake_smaps.append(smap=fake)
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
            weights.append(weight=torch.ones((1, 1, 1, 1)).to(self.device))
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
            weight=self.class_weight_[[ctx.label], :].view(
                1, channel_shape(self.class_weight_), 1, 1
            )
        )
        weights.finalize()
        return weights

    def _clamp_gradient(
        self: ActivationWeight,
        gradient: Tensor,
    ) -> Tensor:
        """clamp inf and -inf on 0.

        Args:
            gradient (Tensor): gradient.

        Returns:
            Tensor: clamped gradient.
        """
        # gradient_nan: np.ndarray = (
        #     (gradient * ~gradient.isinf()).detach().cpu().numpy()
        # )
        # gradient_max: float = np.nanmax(gradient_nan)
        # gradient_min: float = np.nanmin(gradient_nan)
        # return gradient.clamp(min=gradient_min, max=gradient_max)
        return torch.where(gradient.isinf(), 0.0, gradient)

    def _arrange_weight(
        self: ActivationWeight,
        weight: Tensor,
    ) -> Tensor:
        """arrange weight.

        Args:
            layer (int): layer offset (0 == last Conv. Layer).
            weight (Tensor): weight.

        Returns:
            Tensor: arranged weight.
        """
        # gap or not
        if not self.gradient_no_gap_:
            k: int = channel_shape(weight)
            weight = weight.view(1, k, -1).mean(dim=2).view(1, k, 1, 1)
        return weight

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
            # weight = gradient
            weight: Tensor = self._clamp_gradient(gradient=gradient)
            weights.append(weight=self._arrange_weight(weight=weight))
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
            gradient = self._clamp_gradient(gradient=gradient)
            # alpha = gradient ** 2 /
            #         2 * (gradient ** 2) + SUM(avtivation * (gradient ** 3))
            # SUM: sum per channels over positions
            k: int = channel_shape(activation)
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
            weight: Tensor = alpha * gradient
            weights.append(weight=self._arrange_weight(weight=weight))
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
            gradient = self._clamp_gradient(gradient=gradient)
            # weight = gradient * activation
            weight: Tensor = gradient * activation
            weights.append(weight=self._arrange_weight(weight=weight))
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
        weights: Weights = fn(ctx=ctx)
        # create saliency maps by multiplying weight and activation
        smaps: SaliencyMaps = SaliencyMaps()
        for weight, activation in zip(weights, ctx.activations):
            smaps.append(smap=weight * activation)
        smaps.finalize()
        weights.clear()
        return smaps
