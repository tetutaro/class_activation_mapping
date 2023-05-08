#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Callable, Any

import torch
from torch import Tensor
import torch.nn.functional as F

from cam.base import (
    DEBUG,
    batch_shape,
    channel_shape,
    CommonSMAP,
)
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
        # stack
        stacked_smap: Tensor = torch.stack([x for x in channel_smaps], dim=0)
        # sum -> final saliency map
        final_smap_p: Tensor = F.relu(stacked_smap).sum(dim=0)
        final_smap_m: Tensor = -F.relu(-stacked_smap).sum(dim=0)
        final_smap: Tensor = torch.where(
            final_smap_p > self.eps, final_smap_p, final_smap_m
        ).squeeze()
        channel_smaps.clear()
        del stacked_smap
        del final_smap_p
        del final_smap_m
        return final_smap

    def _multiply_channel_smaps(
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
        raise NotImplementedError("not implemented merge_layer (multiply)")
        return

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
        # enlarge channel saliency maps
        enlarged_smaps: SaliencyMaps = ctx.enlarge_fn(smaps=channel_smaps)
        # if number of layres == 1, just return channel saliency map
        if len(enlarged_smaps) == 1:
            return enlarged_smaps[0].squeeze()
        # merge channel saliency maps over layers
        fn: Callable[[SaliencyMaps, Context], Tensor]
        if self.merge_layer == "none":
            fn = self._sum_channel_smaps
        elif self.merge_layer == "multiply":
            fn = self._multiply_channel_smaps
        else:
            raise SystemError(f"invalid merge_layer: {self.merge_layer}")
        return fn(channel_smaps=enlarged_smaps, ctx=ctx)
