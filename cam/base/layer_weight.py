#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from cam.base import (
    DEBUG,
    batch_shape,
    channel_shape,
    CommonWeight,
)
from cam.base import position_shape
from cam.base.containers import SaliencyMaps
from cam.base.context import Context


class LayerWeight(CommonWeight):
    """A part of the CAM model that is responsible for final saliency map.

    XXX

    Args:
        high_resolution (bool): if True, produce high resolution heatmap.
    """

    def __init__(
        self: LayerWeight,
        high_resolution: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # store flags
        self.high_resolution_: bool = high_resolution
        return

    def _high_resolution(
        self: LayerWeight,
        smap: Tensor,
        activation: Tensor,
        factor: int,
    ) -> Tensor:
        """increase resolution of saliency map
        using activation as detailed mask.

        Args:
            smap (Tensor): the saliency map.
            activation (Tensor): activation.
            factor (int): scale factor.

        Returns:
            Tensor: saliency map which is increased its resolution.
        """
        mask_p: Tensor = F.relu(activation).max(dim=1, keepdim=True).values
        mask_m: Tensor = F.relu(-activation).max(dim=1, keepdim=True).values
        # blur mask
        mask_p = mask_p / (
            F.interpolate(
                F.avg_pool2d(mask_p, kernel_size=factor),
                scale_factor=factor,
                mode="bilinear",
            )
            + self.eps
        )
        mask_m = mask_m / (
            F.interpolate(
                F.avg_pool2d(mask_m, kernel_size=factor),
                scale_factor=factor,
                mode="bilinear",
            )
            + self.eps
        )
        final_smap = (mask_p * F.relu(smap)) - (mask_m * F.relu(-smap))
        return final_smap

    def weight_layer(
        self: LayerWeight,
        smaps: SaliencyMaps,
        ctx: Context,
    ) -> Tensor:
        """merge layer saliency maps and conver it to heatmap.

        Args:
            channel_smaps (SaliencyMaps): channel saliency maps
            ctx (Context): the context of this process.

        Returns:
            Tensor: final saliency map.
        """
        n_layers: int = len(smaps)
        if DEBUG:
            assert n_layers > 0
            for smap in smaps:
                assert batch_shape(smap) == 1
                assert channel_shape(smap) == 1
        # if number of layres == 1, just return channel saliency map
        if n_layers == 1:
            # enlarge channel saliency maps
            return ctx.enlarge_one_fn(smap=smaps[0]).squeeze()
        # prepare the saliency map of the last layer
        final_smap: Tensor = smaps[0]
        base_u, base_v = position_shape(ctx.activations[0])
        if not position_shape(final_smap) == (base_u, base_v):
            # downsampling
            final_smap = F.interpolate(
                final_smap,
                size=(base_u, base_v),
                mode="bilinear",
            )
        # normalize
        final_smap = (final_smap / final_smap.max()).clamp(min=-1.0, max=1.0)
        # merge layers
        for layer in range(1, n_layers):
            # prepare saliency map of current layer
            acti: Tensor = ctx.activations[layer]
            smap: Tensor = smaps[layer]
            u, v = position_shape(acti)
            if DEBUG:
                assert u % base_u == 0
                assert u > base_u
                assert (u // base_u) * base_v == v
            if not position_shape(smap) == (u, v):
                # downsampling
                smap = F.interpolate(
                    smap,
                    size=(u, v),
                    mode="bilinear",
                )
            # normalize
            smap = (smap / smap.max()).clamp(min=-1.0, max=1.0)
            # scale factor
            factor: int = u // base_u
            # upsampling base saliency map
            final_smap = F.interpolate(
                final_smap,
                scale_factor=factor,
                mode="bilinear",
            )
            final_smap = (final_smap / final_smap.max()).clamp(min=-1, max=1)
            if self.high_resolution_:
                # high resolution
                final_smap = self._high_resolution(
                    smap=final_smap,
                    activation=acti,
                    factor=factor,
                )
            else:
                # take max between base saliency map and current saliency map
                final_smap_p = (
                    torch.stack([F.relu(final_smap), F.relu(smap)], dim=0)
                    .max(dim=0)
                    .values
                )
                final_smap_m = (
                    torch.stack([F.relu(-final_smap), F.relu(-smap)], dim=0)
                    .max(dim=0)
                    .values
                )
                final_smap = final_smap_p - final_smap_m
            base_u = u
            base_v = v
        smaps.clear()
        # enlarge final saliency map to the original image size
        return ctx.enlarge_one_fn(smap=final_smap).squeeze()
