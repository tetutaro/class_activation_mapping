#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Callable, NamedTuple

from PIL import Image
from torch import Tensor
import torch.nn.functional as F

from cam.base import position_shape
from cam.base.containers import Weights, SaliencyMaps


class Context(NamedTuple):
    """the context of CAM model"""

    forward_fn: Callable[[Tensor], Tensor]
    classify_fn: Callable[[Tensor], Tensor]
    raw_image: Image
    width: int
    height: int
    image: Tensor
    blurred_image: Tensor
    label: int
    score: float
    activations: Weights
    gradients: Weights

    def clear(self: Context) -> None:
        """clear this context."""
        self.activations.clear()
        self.gradients.clear()
        return

    def enlarge_fn(
        self: Context,
        smaps: SaliencyMaps,
    ) -> SaliencyMaps:
        """enlarge saliency maps to the original image size.

        Args:
            smaps (SaliencyMaps): the saliency maps to enlarge.

        Returns:
            SaliencyMaps: enlarged saliency maps.
        """
        enlarged_smaps: SaliencyMaps = SaliencyMaps()
        for smap in smaps:
            u, v = position_shape(smap)
            if u == self.height and v == self.width:
                enlarged_smaps.append(smap=smap.clone())
            else:
                enlarged_smaps.append(
                    smap=F.interpolate(
                        smap,
                        size=(self.height, self.width),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
        enlarged_smaps.finalize()
        smaps.clear()
        return enlarged_smaps
