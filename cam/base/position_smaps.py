#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Optional, Callable, Any

import torch
from torch import Tensor

from cam.base import DEBUG, batch_shape, CommonSMAP
from cam.base.containers import SaliencyMaps
from cam.base.context import Context


class PositionSMAPS(CommonSMAP):
    """A part of the CAM model that is responsible for position saliency maps.

    XXX
    """

    def __init__(self: PositionSMAPS, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # store flags
        self.position_weight: str = kwargs["position_weight"]
        self.position_group: str = kwargs["position_group"]
        self.n_positions_: int = kwargs["n_positions"]
        self.n_position_groups: Optional[int] = kwargs["n_position_groups"]
        return

    def _fake_smaps(
        self: PositionSMAPS,
        raw_smaps: SaliencyMaps,
    ) -> SaliencyMaps:
        """Fake-CAM

        S. Poppi, et al.
        "Revisiting The Evaluation of Class Activation Mapping
        for Explainability:
        A Novel Metric and Experimental Analysis"
        CVPR 2021.

        https://arxiv.org/abs/2104.10252

        Args:
            raw_smaps (SaliencyMaps): raw saliency maps.

        Returns:
            SaliencyMaps: fake saliency maps
        """
        fake_smaps = SaliencyMaps()
        for smap in raw_smaps:
            fake: Tensor = torch.ones_like(smap)
            fake[:, :, 0, 0] = 0.0  # set 0 to the top-left corner
            fake_smaps.append(fake)
        fake_smaps.finalize()
        raw_smaps.clear()
        return fake_smaps

    def _position_eigen_weight(
        self: PositionSMAPS,
        smap: Tensor,
        ctx: Context,
    ) -> Tensor:
        raise NotImplementedError(
            f"invalid position_weight {self.position_weight}"
        )
        return

    def _position_ablation_weight(
        self: PositionSMAPS,
        smap: Tensor,
        ctx: Context,
    ) -> Tensor:
        raise NotImplementedError(
            f"invalid position_weight {self.position_weight}"
        )
        return

    def _position_abscission_weight(
        self: PositionSMAPS,
        smap: Tensor,
        ctx: Context,
    ) -> Tensor:
        raise NotImplementedError(
            f"invalid position_weight {self.position_weight}"
        )
        return

    def create_position_smaps(
        self: PositionSMAPS,
        raw_smaps: SaliencyMaps,
        ctx: Context,
    ) -> SaliencyMaps:
        """create position saliency maps from gradient saliency maps.

        Args:
            gradient_smaps (SaliencyMaps): gradient saliency maps
            ctx (Context): the context of this process.

        Returns:
            SaliencyMaps: position saliency maps.
        """
        # special cases
        if self.position_weight == "none":
            return raw_smaps
        if self.position_weight == "fake":
            return self._fake_smaps(raw_smaps=raw_smaps)
        # forcibly enlarge each saliency maps to the original image size
        enlarged_smaps: SaliencyMaps = ctx.enlarge_fn(smaps=raw_smaps)
        # create the weight for each positions
        # and create position saliency map for each layers
        position_smaps: SaliencyMaps = SaliencyMaps()
        for smap in enlarged_smaps:
            if DEBUG:
                assert batch_shape(smap) == 1
            # the function to create weights for each positions
            fn: Callable[[Tensor, Context], Tensor]
            if self.position_weight in ["eigen", "cosine"]:
                fn = self._position_eigen_weight
            elif self.position_weight == "ablation":
                fn = self._position_ablation_weight
            elif self.position_weight == "abscission":
                fn = self._position_abscission_weight
            else:
                raise SystemError(
                    f"invalid position_weight: {self.position_weight}"
                )
            # create the weight and multiply it to raw saliency map
            position_smaps.append(smap=fn(smap=smap, ctx=ctx) * smap)
        position_smaps.finalize()
        enlarged_smaps.clear()
        return position_smaps
