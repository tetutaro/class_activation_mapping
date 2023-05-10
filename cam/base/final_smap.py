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
    CommonSMAP,
)
from cam.base import position_shape
from cam.base.containers import SaliencyMaps
from cam.base.context import Context


class FinalSMAP(CommonSMAP):
    """A part of the CAM model that is responsible for final saliency map.

    XXX
    """

    def __init__(self: FinalSMAP, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # store flags
        self.merge_layer: str = kwargs["merge_layer"]
        return

    def _sum_channel_smaps(
        self: FinalSMAP,
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
        final_smap: Tensor = stacked_smap.sum(dim=0).squeeze()
        channel_smaps.clear()
        del stacked_smap
        return final_smap

    def _mul_channel_smaps(
        self: FinalSMAP,
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
        smap_list = [x for x in channel_smaps]
        final_smap: Tensor = smap_list[0]
        base_u, base_v = position_shape(final_smap)
        for smap in smap_list[1:]:
            u, v = position_shape(smap)
            if DEBUG:
                assert u % base_u == 0
                assert u > base_u
                assert (u // base_u) * base_v == v
            factor: int = u // base_u
            # create blurred mask from activation
            mask: Tensor = smap / (
                F.interpolate(
                    F.avg_pool2d(F.relu(smap), kernel_size=factor),
                    scale_factor=factor,
                    mode="bilinear",
                )
                + self.eps
            )
            # update final saliency map by multiplying mask
            final_smap = mask * F.interpolate(
                final_smap,
                scale_factor=factor,
                mode="bilinear",
            )
            base_u = u
            base_v = v
        channel_smaps.clear()
        # enlarge final saliency map to the original image size
        return F.interpolate(
            final_smap,
            size=(ctx.height, ctx.width),
            mode="bilinear",
        ).squeeze()

    def create_final_smap(
        self: FinalSMAP,
        channel_smaps: SaliencyMaps,
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
            assert len(channel_smaps) > 0
            for smap in channel_smaps:
                assert batch_shape(smap) == 1
                assert channel_shape(smap) == 1
        # if number of layres == 1, just return channel saliency map
        if len(channel_smaps) == 1:
            # enlarge channel saliency maps
            return ctx.enlarge_fn(smaps=channel_smaps)[0].squeeze()
        # merge channel saliency maps over layers
        fn: Callable[[SaliencyMaps, Context], Tensor]
        if self.merge_layer == "none":
            fn = self._sum_channel_smaps
        elif self.merge_layer == "multiply":
            fn = self._mul_channel_smaps
        else:
            raise SystemError(f"invalid merge_layer: {self.merge_layer}")
        return fn(channel_smaps=channel_smaps, ctx=ctx)
