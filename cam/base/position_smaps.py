#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Optional, Callable, Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from cam.base import (
    DEBUG,
    batch_shape,
    channel_shape,
    position_shape,
    CommonSMAP,
)
from cam.base.containers import SaliencyMaps
from cam.base.context import Context
from cam.base.groups import group_kmeans, group_spectral, weight_minmax


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
        self.n_position_groups_: Optional[int] = kwargs["n_position_groups"]
        self.position_minmax_: bool = kwargs["position_minmax"]
        if self.position_weight == "none":
            self.position_minmax_ = False
        return

    def _fake_smaps(
        self: PositionSMAPS,
        raw_smaps: SaliencyMaps,
    ) -> SaliencyMaps:
        """create fake saliency map whose values are almost 1.
        (except the top left corner)

        Args:
            raw_smaps (SaliencyMaps): raw saliency maps.

        Returns:
            SaliencyMaps: fake saliency maps.
        """
        fake_smaps = SaliencyMaps()
        for smap in raw_smaps:
            fake: Tensor = torch.ones_like(smap)
            fake[:, :, 0, 0] = 0.0  # set 0 to the top-left corner
            fake_smaps.append(fake)
        fake_smaps.finalize()
        raw_smaps.clear()
        return fake_smaps

    # ## functions to create position group map (position -> position group)

    def _position_group_none(
        self: PositionSMAPS,
        smap: Tensor,
        ctx: Context,
    ) -> Tensor:
        """dummy clustering.

        Args:
            smap (Tensor): the saliemcy map (1 x channel x height x width).
            ctx (Context): the context of this process.

        Returns:
            Tensor:
                identity matrix (position x position).
                (position = height x width)
        """
        u, v = position_shape(smap)
        return torch.diag(torch.ones(u * v)).to(self.device)

    def _position_group_kmeans(
        self: PositionSMAPS,
        smap: Tensor,
        ctx: Context,
    ) -> Tensor:
        """clutering positions using k-Means.

        Args:
            smap (Tensor): the saliemcy map (1 x channel x height x width).
            ctx (Context): the context of this process.

        Returns:
            Tensor:
                position group map (group x position).
                (position = height x width)
        """
        # create features to create position group map
        k: int = channel_shape(smap)
        features: np.ndarray = smap.view(k, -1).T.detach().cpu().numpy()
        # create position group map using k-Means
        gmap: Tensor = group_kmeans(
            features=features,
            n_groups=self.n_position_groups_,
            random_state=self.random_state,
        ).to(self.device)
        if self.n_position_groups_ is None:
            # store the optimal number of groups in self.n_channel_groups_
            self.n_position_groups_ = batch_shape(gmap)
        return gmap

    def _position_group_spectral(
        self: PositionSMAPS,
        smap: Tensor,
        ctx: Context,
    ) -> Tensor:
        """clutering positions using Spectral Clustering.

        Args:
            smap (Tensor): the saliemcy map (1 x channel x height x width).
            ctx (Context): the context of this process.

        Returns:
            Tensor:
                position group map (group x position).
                (position = height x width)
        """
        # create features to create position group map
        k: int = channel_shape(smap)
        features: np.ndarray = smap.view(k, -1).T.detach().cpu().numpy()
        # create channel group map using Spectral Clustering
        gmap: Tensor = group_spectral(
            features=features,
            n_groups=self.n_position_groups_,
            random_state=self.random_state,
        ).to(self.device)
        if self.n_position_groups_ is None:
            # store the optimal number of groups in self.n_channel_groups_
            self.n_position_groups_ = batch_shape(gmap)
        return gmap

    # ## functions to create weight for each position group

    def _position_weight_none(
        self: PositionSMAPS,
        smap: Tensor,
        gmap: Tensor,
        ctx: Context,
    ) -> Tensor:
        """weight by the number of positions for each position groups.

        Args:
            smap (Tensor): the saliency map (1 x channel x height x width).
            gmap (Tensor): the position group map (group x position).
            ctx (Context): the context of this process.

        Returns:
            Tensor: weight for each position groups.
        """
        return torch.where(gmap > 0, 1.0, 0.0).sum(dim=1) / channel_shape(gmap)

    def _position_weight_eigen(
        self: PositionSMAPS,
        smap: Tensor,
        gmap: Tensor,
        ctx: Context,
    ) -> Tensor:
        """the first eigen-vector of the position space.

        the saliency map is divided into
        the position space and the channel space by
        SVD (Singular Value Decomposition).
        use the first eigen-vector of the position space
        as the weight for each channel groups.

        Args:
            smap (Tensor): the saliency map (1 x channel x height x width).
            gmap (Tensor): the position group map (group x position).
            ctx (Context): the context of this process.

        Returns:
            Tensor: weights for each position groups.
        """
        _, k, u, v = smap.shape
        # create group x channel matrix
        GC: Tensor = gmap @ smap.view(k, -1).T
        # standarize
        GC_std: Tensor = (GC - GC.mean()) / GC.std()
        # SVD (singular value decomposition)
        # Gs = position group space, ss = eigen-values
        Gs, ss, _ = torch.linalg.svd(GC_std, full_matrices=False)
        # retrieve the first eigen-vector and normalize it
        weight: Tensor = F.normalize(Gs.real[:, ss.argmax()], dim=0)
        # test the eigen-vector may have the opposite sign
        if (
            (weight.view(1, -1) @ gmap).view(1, 1, u, v) * F.relu(smap)
        ).sum() < 0:
            weight = -weight
        return weight

    def _position_weight_ablation(
        self: PositionSMAPS,
        smap: Tensor,
        gmap: Tensor,
        ctx: Context,
    ) -> Tensor:
        """the first eigen-vector of the position space.

        the saliency map is divided into
        the position group space and the channel space by
        SVD (Singular Value Decomposition).
        Each vertical vectors in the position group space
        is the eigen-vector of the position group space.
        use the first eigen-vector (that has the highest eigen-value)
        of the position group space as the weight for each position groups.

        Args:
            smap (Tensor): the saliency map (1 x channel x height x width).
            gmap (Tensor): the position group map (group x position).
            ctx (Context): the context of this process.

        Returns:
            Tensor: weights for each position groups.
        """
        if DEBUG:
            len(ctx.activations) == 1
        activation: Tensor = ctx.activations[0]
        u, _ = position_shape(activation)
        g: int = batch_shape(gmap)
        ablation_list: List[Tensor] = list()
        for i in range(g):
            ablation: Tensor = activation.clone()
            # group mask
            mask_indexes: Tensor = torch.where(gmap[i, :].view(-1) > 0)[0]
            # drop i-th position group (ablation)
            for mask_index in mask_indexes:
                mask_u = mask_index // u
                mask_v = mask_index % u
                ablation[:, :, mask_u, mask_v] = 0
            ablation_list.append(ablation)
        ablations: Tensor = torch.cat(ablation_list, dim=0)
        # forward Ablations and retrieve Ablation score
        a_scores: Tensor = ctx.classify_fn(activation=ablations)[:, ctx.label]
        # slope of Ablation score
        return (ctx.score - a_scores) / (ctx.score + self.eps)

    def _position_weight_abscission(
        self: PositionSMAPS,
        smap: Tensor,
        gmap: Tensor,
        ctx: Context,
    ) -> Tensor:
        """PIC (Position-wise Increase of Confidence) scores.

        Args:
            smap (Tensor): the saliency map (1 x channel x height x width).
            gmap (Tensor): the group map (group x position).
            ctx (Context): the context of this process.

        Returns:
            Tensor: weights for each position groups.
        """
        # create grouped saliency map
        u, v = position_shape(smap)
        g: int = batch_shape(gmap)
        mask_dicts: List[Dict] = list()
        for i in range(g):
            # position group mask for abscission
            abs_mask: Tensor = torch.zeros(1, 1, u, v).to(self.device)
            abs_mask_idxes: Tensor = torch.where(gmap[i, :].view(-1) > 0)[0]
            for abs_mask_idx in abs_mask_idxes:
                abs_mask_u = abs_mask_idx // u
                abs_mask_v = abs_mask_idx % u
                abs_mask[:, :, abs_mask_u, abs_mask_v] = 1.0
            # extract i-th group from the saliency map ("Abscission")
            abscission: Tensor = abs_mask * (
                (smap - smap.min()) / (smap.max() - smap.min() + self.eps)
            ).mean(dim=1, keepdim=True)
            # normalize
            smax: Tensor = abscission.max().squeeze()
            smin: Tensor = abscission.min().squeeze()
            sdif: Tensor = smax - smin
            if sdif < self.eps:
                continue
            # smoother mask for the original image
            mask: Tensor = (abscission - smin) / sdif
            # stack information
            mask_dicts.append(
                {
                    "group": i,
                    "key": sdif.detach().cpu().numpy().ravel()[0],
                    "mask": mask,
                }
            )
        if self.n_positions_ > 0:
            mask_dicts = sorted(
                mask_dicts, key=lambda x: x["key"], reverse=True
            )[: self.n_positions_]
        # create masked image
        masked_list: List[Tensor] = list()
        for mask_dict in mask_dicts:
            masked: Tensor = ctx.image * mask_dict["mask"]
            # SIGCAM
            masked += ctx.blurred_image * (1.0 - mask_dict["mask"])
            masked_list.append(masked)
        maskedes: Tensor = torch.cat(masked_list, dim=0)
        # forward network then retrieve reduced score
        reduced_scores: Tensor = ctx.forward_fn(image=maskedes)[:, ctx.label]
        del maskedes
        # calc CIC score and normalize it
        cic_scores = F.normalize(reduced_scores - ctx.score, dim=0)
        # create weight
        weight: Tensor
        if batch_shape(cic_scores) < g:
            idx: Tensor = torch.tensor([x["group"] for x in mask_dicts]).to(
                self.device
            )
            weight = (
                torch.zeros(g)
                .scatter(dim=0, index=idx, src=cic_scores)
                .to(self.device)
            )
            del cic_scores
        else:
            weight = cic_scores
        return weight

    # ## main function

    def create_position_smaps(
        self: PositionSMAPS,
        raw_smaps: SaliencyMaps,
        ctx: Context,
    ) -> SaliencyMaps:
        """create position saliency maps from raw saliency maps.

        Args:
            raw_smaps (SaliencyMaps): the raw saliency maps.
            ctx (Context): the context of this process.

        Returns:
            SaliencyMaps: position saliency maps.
        """
        # special cases
        if self.position_weight == "none" and self.position_group == "none":
            return raw_smaps
        if self.position_weight == "fake":
            return self._fake_smaps(raw_smaps=raw_smaps)
        if self.position_weight == "abscission":
            # forcibly enlarge saliency maps to the original image size
            raw_smaps = ctx.enlarge_fn(smaps=raw_smaps)
        # create position saliency map for each layers
        position_smaps: SaliencyMaps = SaliencyMaps()
        for smap in raw_smaps:
            b, k, u, v = smap.size()
            if DEBUG:
                assert b == 1
            # the function to create position group map
            group_fn: Callable[[Tensor, Context], Tensor]
            if self.position_group == "none":
                group_fn = self._position_group_none
            elif self.position_group == "k-means":
                group_fn = self._position_group_kmeans
            elif self.position_group == "spectral":
                group_fn = self._position_group_spectral
            else:
                raise SystemError(
                    f"invalid position_group: {self.position_group}"
                )
            # the function to create weights for each position groups
            weight_fn: Callable[[Tensor, Tensor, Context], Tensor]
            if self.position_weight == "eigen":
                weight_fn = self._position_weight_eigen
            elif self.position_weight == "ablation":
                weight_fn = self._position_weight_ablation
            elif self.position_weight == "abscission":
                weight_fn = self._position_weight_abscission
            else:
                raise SystemError(
                    f"invalid position_weight: {self.position_weight}"
                )
            # position group map
            gmap: Tensor = group_fn(smap=smap, ctx=ctx)
            if DEBUG:
                assert channel_shape(gmap) == u * v
                assert torch.allclose(
                    gmap.sum(dim=1), torch.ones(batch_shape(gmap))
                )
            # create weight for each position groups
            weight: Tensor = weight_fn(smap=smap, gmap=gmap, ctx=ctx)
            if self.position_minmax_:
                # cognition-base and cognition-scissors
                weight = weight_minmax(weight=weight).to(self.device)
            weight = (weight.view(1, -1) @ gmap).view(1, 1, u, v)
            # weight saliency map by its position
            position_smaps.append(smap=weight * smap)
        # finalize
        position_smaps.finalize()
        raw_smaps.clear()
        return position_smaps
