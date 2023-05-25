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
        final_smap: Tensor,
        activation: Tensor,
        factor: int,
    ) -> Tensor:
        """increase resolution of saliency map
        using activation as detailed mask.

        Args:
            final_smap (Tensor): the saliency map.
            activation (Tensor): activation.
            factor (int): scale factor.

        Returns:
            Tensor: saliency map which is increased its resolution.
        """
        # positive and negative mask
        mask_p: Tensor = F.relu(activation).max(dim=1, keepdim=True).values
        mask_m: Tensor = F.relu(-activation).max(dim=1, keepdim=True).values
        # blur mask
        mask_p = mask_p / (
            F.interpolate(
                F.avg_pool2d(mask_p, kernel_size=factor),
                scale_factor=factor,
                mode="bilinear",
                align_corners=False,
            )
            + self.eps
        )
        mask_m = mask_m / (
            F.interpolate(
                F.avg_pool2d(mask_m, kernel_size=factor),
                scale_factor=factor,
                mode="bilinear",
                align_corners=False,
            )
            + self.eps
        )
        # mask saliency map
        final_smap = (mask_p * F.relu(final_smap)) - (
            mask_m * F.relu(-final_smap)
        )
        return final_smap

    def _merge_saliency_maps(
        self: LayerWeight,
        final_smap: Tensor,
        smap: Tensor,
    ) -> Tensor:
        """merge base saliency map and current saliency map
        using base saliency map as mask.

        Args:
            final_smap (Tensor): base saliency map.
            smap (Tensor): current saliency map.

        Returns:
            Tensor: merged saliency map.
        """
        # positive and negative mask
        mask_p: Tensor = torch.where(final_smap > 0, 1.0, 0.0)
        mask_m: Tensor = torch.where(final_smap < 0, 1.0, 0.0)
        # take max between base saliency map and current saliency map
        final_smap_p = (
            torch.stack([F.relu(final_smap), F.relu(mask_p * smap)], dim=0)
            .max(dim=0)
            .values
        )
        final_smap_m = (
            torch.stack([F.relu(-final_smap), F.relu(-(mask_m * smap))], dim=0)
            .max(dim=0)
            .values
        )
        final_smap = final_smap_p - final_smap_m
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
        if DEBUG:
            assert position_shape(final_smap) == (base_u, base_v)
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
                assert position_shape(smap) == (u, v)
            # normalize current saliency map
            smap = (smap / smap.max()).clamp(min=-1.0, max=1.0)
            # scale factor
            factor: int = u // base_u
            # upsampling base saliency map
            final_smap = F.interpolate(
                final_smap,
                scale_factor=factor,
                mode="bilinear",
                align_corners=False,
            )
            # merge
            if self.high_resolution_:
                # high resolution
                final_smap = self._high_resolution(
                    final_smap=final_smap,
                    activation=acti,
                    factor=factor,
                )
            else:
                # merge saliency map simply
                final_smap = self._merge_saliency_maps(
                    final_smap=final_smap,
                    smap=smap,
                )
            base_u = u
            base_v = v
        smaps.clear()
        # enlarge final saliency map to the original image size
        return ctx.enlarge_one_fn(smap=final_smap).squeeze()
