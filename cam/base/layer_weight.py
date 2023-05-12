#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Callable, Any

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
    """

    def __init__(self: LayerWeight, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # store flags
        self.high_resolution_: bool = kwargs["high_resolution"]
        return

    def _sum_channel_smaps(
        self: LayerWeight,
        channel_smaps: SaliencyMaps,
        ctx: Context,
    ) -> Tensor:
        """Layer-CAM

        Args:
            channel_smaps (SaliencyMaps): channel saliency maps.
            ctx (Context): the context of this process.

        Returns:
            Tensor: final saliency map.
        """
        # enlarge
        enlarged: SaliencyMaps = ctx.enlarge_fn(smaps=channel_smaps)
        # normalize
        smap_list: List[Tensor] = list()
        for smap in enlarged:
            smap_list.append(smap / smap.max())
        stacked_smap: Tensor = torch.stack(smap_list, dim=0)
        # sum -> final saliency map
        final_smap: Tensor = (
            F.relu(stacked_smap).sum(dim=0) - F.relu(-stacked_smap).sum(dim=0)
        ).squeeze()
        channel_smaps.clear()
        del stacked_smap
        return final_smap

    def _mul_channel_smaps(
        self: LayerWeight,
        channel_smaps: SaliencyMaps,
        ctx: Context,
    ) -> Tensor:
        """Poly-CAM

        Args:
            channel_smaps (SaliencyMaps): channel saliency maps.
            ctx (Context): the context of this process.

        Returns:
            Tensor: final saliency map.
        """
        # base of final saliency map
        n_layers: int = len(channel_smaps)
        final_smap: Tensor = channel_smaps[0]
        base_u, base_v = position_shape(final_smap)
        for layer in range(1, n_layers):
            # create mask from activation
            activation: Tensor = ctx.activations[layer]
            mask_p: Tensor = F.relu(activation).max(dim=1, keepdim=True).values
            mask_m: Tensor = (
                F.relu(-activation).max(dim=1, keepdim=True).values
            )
            mask: Tensor = mask_p - mask_m
            u, v = position_shape(mask)
            if DEBUG:
                assert u % base_u == 0
                assert u > base_u
                assert (u // base_u) * base_v == v
                assert mask.shape == channel_smaps[layer].shape
            factor: int = u // base_u
            # normalize mask
            # m_max: Tensor = mask.max()
            # m_min: Tensor = mask.min()
            # mask = (mask - m_min) / (m_max - m_min + self.eps)
            # mask = (mask / mask.max()).clamp(min=-1.0, max=1.0)
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
            # update final saliency map by multiplying mask
            f_smap = F.interpolate(
                final_smap,
                scale_factor=factor,
                mode="bilinear",
            )
            final_smap = (mask_p * F.relu(f_smap)) - (mask_m * F.relu(-f_smap))
            base_u = u
            base_v = v
        channel_smaps.clear()
        # enlarge final saliency map to the original image size
        return F.interpolate(
            final_smap,
            size=(ctx.height, ctx.width),
            mode="bilinear",
        ).squeeze()

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
        if DEBUG:
            assert len(smaps) > 0
            for smap in smaps:
                assert batch_shape(smap) == 1
                assert channel_shape(smap) == 1
        # if number of layres == 1, just return channel saliency map
        if len(smaps) == 1:
            # enlarge channel saliency maps
            return ctx.enlarge_fn(smaps=smaps)[0].squeeze()
        # merge channel saliency maps over layers
        fn: Callable[[SaliencyMaps, Context], Tensor]
        if self.high_resolution_:
            fn = self._mul_channel_smaps
        else:
            fn = self._sum_channel_smaps
        return fn(channel_smaps=smaps, ctx=ctx)
