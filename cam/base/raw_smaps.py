#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Callable, Any

import torch
from torch import Tensor

from cam.base import channel_shape, CommonSMAP
from cam.base.containers import Weights, SaliencyMaps
from cam.base.context import Context


class RawSMAPS(CommonSMAP):
    """A part of the CAM model that is responsible for raw saliency maps.

    RawSMAPS (output of Conv. Layer(s) when forwarding) represent
    which region of image was payed attention to recognize the target object.

    XXX
    """

    def __init__(self: RawSMAPS, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # store flags
        self.activation_weight: str = kwargs["activation_weight"]
        self.gradient_gap_: bool = kwargs["gradient_gap"]
        self.gradient_smooth: str = kwargs["gradient_smooth"]
        if self.activation_weight in ["none", "class"]:
            self.gradient_gap_ = False
        self.class_weights_: Tensor
        return

    def set_class_weights(self: RawSMAPS, class_weights: Tensor) -> None:
        """set class weights

        Args:
            class_weights (Tensor): class weights
        """
        self.class_weights_ = class_weights
        return

    def _dummy_weights(self: RawSMAPS, ctx: Context) -> Weights:
        """returns the dummy weights (each values of weights = 1).

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

    def _class_weights(self: RawSMAPS, ctx: Context) -> Weights:
        """CAM

        B. Zhou, et al.
        "Learning Deep Features for Discriminative Localization"
        CVPR 2016.

        https://arxiv.org/abs/1512.04150

        Args:
            ctx (Context): the context of this process.

        Returns:
            Weights: the weight retrieve from weights of the classifier part.
        """
        weights: Weights = Weights()
        weights.append(
            self.class_weights_[[ctx.label], :].view(
                1, channel_shape(self.class_weights_), 1, 1
            )
        )
        weights.finalize()
        return weights

    def _grad_weights(self: RawSMAPS, ctx: Context) -> Weights:
        """Grad-CAM

        R. Selvaraju, et al.
        "Grad-CAM:
        Visual Explanations from Deep Networks via Gradient-based Localization"
        ICCV 2017.

        https://arxiv.org/abs/1610.02391

        Args:
            ctx (Context): the context of this process.

        Returns:
            Weights: gradients.
        """
        weights: Weights = Weights()
        for gradient in ctx.gradients:
            weights.append(gradient.clone())
        weights.finalize()
        return weights

    def _grad_pp_weights(self: RawSMAPS, ctx: Context) -> Weights:
        """Grad-CAM++

        A. Chattopadhyay, et al.
        "Grad-CAM++:
        Improved Visual Explanations for Deep Convolutional Networks"
        WACV 2018.

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

    def _grad_axiom_weights(self: RawSMAPS, ctx: Context) -> Weights:
        """XGrad-CAM

        R. Fu, et al.
        "Axiom-based Grad-CAM:
        Towards Accurate Visualization and Explanation of CNNs"
        BMVC 2020.

        https://arxiv.org/abs/2008.02312

        Args:
            ctx (Context): the context of this process.

        Returns:
            Weights: gradient x activation
        """
        weights: Weights = Weights()
        for activation, gradient in zip(ctx.activations, ctx.gradients):
            weights.append(gradient * activation)
        weights.finalize()
        return weights

    def create_raw_smaps(self: RawSMAPS, ctx: Context) -> SaliencyMaps:
        """
        Args:
            ctx (Context): the context of this process.

        Returns:
            Tuple[Forwarded, SaliencyMaps]: forwarded and gradient saliency map
        """
        # get gradient weights
        fn: Callable[[Context], Weights]
        if self.activation_weight == "none":
            fn = self._dummy_weights
        elif self.activation_weight == "class":
            fn = self._class_weights
        elif self.activation_weight == "gradient":
            fn = self._grad_weights
        elif self.activation_weight == "gradient++":
            fn = self._grad_pp_weights
        elif self.activation_weight == "axiom":
            fn = self._grad_axiom_weights
        else:
            raise NotImplementedError(
                f"activation weight is invalid: {self.activation_weight}"
            )
        source: Weights = fn(ctx=ctx)
        # GAP (global average pooling) of gradient weights
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
            # HiRes-CAM
            # R. Draelos, et al.
            # "Use HiResCAM instead of Grad-CAM
            #  for faithful explanations of convolutional neural networks"
            # arXiv 2020.
            weights = source
        # create raw saliency maps by multiplying weights and activations
        raw_smaps: SaliencyMaps = SaliencyMaps()
        for weight, activation in zip(weights, ctx.activations):
            raw_smaps.append(smap=weight * activation)
        raw_smaps.finalize()
        weights.clear()
        return raw_smaps
