#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Optional, Callable, Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from sklearn.cluster import KMeans, SpectralClustering
from kneed import KneeLocator

from cam.base import (
    DEBUG,
    batch_shape,
    channel_shape,
    position_shape,
    CommonSMAP,
)
from cam.base.containers import SaliencyMaps
from cam.base.context import Context


class ChannelSMAPS(CommonSMAP):
    """A part of the CAM model that is responsible for channel saliency maps.

    XXX
    """

    def __init__(self: ChannelSMAPS, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # store flags
        self.channel_weight: str = kwargs["channel_weight"]
        self.channel_group: str = kwargs["channel_group"]
        self.n_channels_: int = kwargs["n_channels"]
        self.n_channel_groups_: Optional[int] = kwargs["n_channel_groups"]
        self.channel_minmax_: bool = kwargs["channel_minmax"]
        self.random_state_: Optional[int] = kwargs["random_state"]
        if self.channel_weight == "none":
            self.channel_minmax_ = False
        return

    # ## group channels

    def _channel_none_group(
        self: ChannelSMAPS,
        smap: Tensor,
        ctx: Context,
    ) -> Tensor:
        """dummy clustering

        Args:
            smap (Tensor): saliemcy map
            ctx (Context): the context of this process.

        Returns:
            Tensor: (channel x channel) identity matrix
        """
        return torch.diag(torch.ones(channel_shape(smap))).to(self.device)

    def _estimate_n_groups(
        self: ChannelSMAPS,
        features: np.ndarray,
    ) -> int:
        """automatically estimate optimal number of groups.

        estimate optimal number of groups from the elbow point of
        summed squared error of k-Means (inertia).

        Args:
            features (np.ndarray): data.

        Returns:
            int: the optimal number groups
        """
        n_data: int = features.shape[0]
        max_groups: int = int(np.ceil(np.log(n_data))) + 1
        n_groups_list: List[int] = list(range(2, max_groups))
        inertia_list: List[float] = list()
        for n_groups in n_groups_list:
            km: KMeans = KMeans(
                n_clusters=n_groups,
                init="k-means++",
                n_init="auto",
                random_state=self.random_state_,
            ).fit(features)
            inertia_list.append(km.inertia_)
        # retrieve elbow point by KneeLocator
        kneedle: KneeLocator = KneeLocator(
            n_groups_list,
            inertia_list,
            curve="convex",
            direction="decreasing",
        )
        n_groups: int
        if kneedle.elbow is None:
            n_groups = max_groups
        else:
            n_groups = kneedle.elbow
        return n_groups

    def _channel_kmeans_group(
        self: ChannelSMAPS,
        smap: Tensor,
        ctx: Context,
    ) -> Tensor:
        """clustering channels using k-Means

        Args:
            smap (Tensor): saliemcy map
            ctx (Context): the context of this process.

        Returns:
            Tensor: group mappings (n_groups x n_channels)
        """
        k: int = channel_shape(smap)
        features: np.ndarray = smap.view(k, -1).detach().cpu().numpy()
        n_groups: int
        if self.n_channel_groups_ is not None:
            n_groups = self.n_channel_groups_
        else:
            n_groups = self._estimate_n_groups(features=features)
            self.n_channel_groups_ = n_groups
        km: KMeans = KMeans(
            n_clusters=n_groups,
            init="k-means++",
            n_init="auto",
            random_state=self.random_state_,
        ).fit(features)
        group_weight_list: List[List[float]] = list()
        for group in range(n_groups):
            group_weight: np.ndarray = np.where(km.labels_ == group, 1.0, 0.0)
            group_weight /= group_weight.sum()
            group_weight_list.append(group_weight.tolist())
        return torch.tensor(group_weight_list).to(self.device)

    def _channel_spectral_group(
        self: ChannelSMAPS,
        smap: Tensor,
        ctx: Context,
    ) -> Tensor:
        """clustering channels using Spectral Clustering.

        Args:
            smap (Tensor): saliemcy map
            ctx (Context): the context of this process.

        Returns:
            Tensor: group mappings (n_groups x n_channels)
        """
        k: int = channel_shape(smap)
        features: np.ndarray = smap.view(k, -1).detach().cpu().numpy()
        n_groups: int
        if self.n_channel_groups_ is not None:
            n_groups = self.n_channel_groups_
        else:
            n_groups = self._estimate_n_groups(features=features)
            self.n_channel_groups_ = n_groups
        sc: SpectralClustering = SpectralClustering(
            n_clusters=n_groups,
            affinity="nearest_neighbors",
            n_jobs=-1,
            random_state=self.random_state_,
        ).fit(features)
        group_weight_list: List[List[float]] = list()
        for group in range(n_groups):
            group_weight: np.ndarray = np.where(sc.labels_ == group, 1.0, 0.0)
            group_weight /= group_weight.sum()
            group_weight_list.append(group_weight.tolist())
        return torch.tensor(group_weight_list).to(self.device)

    # ## weight channels and merge

    def _channel_none_weight(
        self: ChannelSMAPS,
        smap: Tensor,
        gmap: Tensor,
        ctx: Context,
    ) -> Tensor:
        """global average pooling over channel groups.

        Args:
            smap (Tensor): the saliency map.
            gmap (Tensor): the group mapping (group x channel)
            ctx (Context): the context of this process.

        Returns:
            Tensor: weights for each channel groups.
        """
        g: int = batch_shape(gmap)
        return (torch.ones(g) / g).to(self.device)

    def _channel_eigen_weight(
        self: ChannelSMAPS,
        smap: Tensor,
        gmap: Tensor,
        ctx: Context,
    ) -> Tensor:
        """Eigen-CAM

        M. Muhammad, et al.
        "Eigen-CAM:
        Class Activation Map using Principal Components"
        IJCNN 2020.

        https://arxiv.org/abs/2008.00299

        Args:
            smap (Tensor): the saliency map to calc weight.
            gmap (Tensor): the group mapping (group x channel)
            ctx (Context): the context of this process.

        Returns:
            Tensor: weights for each channel groups.
        """
        k: int = channel_shape(smap)
        u, v = position_shape(smap)
        g: int = batch_shape(gmap)
        # create channel x position matrix
        CP: Tensor = gmap @ smap.view(k, -1)
        if DEBUG:
            batch_shape(CP) == g
            channel_shape(CP) == u * v
        # standarize
        CP_std: Tensor = (CP - CP.mean()) / CP.std()
        # SVD (singular value decomposition)
        Ch, SS, _ = torch.linalg.svd(CP_std, full_matrices=False)
        # retrieve weight vector
        # * Candidate 1: (the same as the Eigen-CAM paper)
        #   the first eigen-vector of the channel matrix
        #     weight: Tensor = Ch.real[:, SS.argmax()]
        # * Candidate 2: projection of each channel vector
        #   to the first eigen-vector of variance-covariance matrix
        #   of the channel matrix
        weight: Tensor = Ch.real[:, SS.argmax()]
        # normalize weight
        weight = F.normalize(weight, dim=0).view(1, g, 1, 1)
        # the eigen-vector may have the opposite sign
        if (weight * F.relu(CP.view(1, g, u, v))).sum() < 0:
            weight = -weight
        return weight.view(-1)

    def _channel_ablation_weight(
        self: ChannelSMAPS,
        smap: Tensor,
        gmap: Tensor,
        ctx: Context,
    ) -> Tensor:
        """Ablation-CAM

        H. Ramaswamy, et al.
        "Ablation-CAM:
        Visual Explanations for Deep Convolutional Network
        via Gradient-free Localization"
        WACV 2020.

        https://openaccess.thecvf.com/content_WACV_2020/html/Desai_Ablation-CAM_Visual_Explanations_for_Deep_Convolutional_Network_via_Gradient-free_Localization_WACV_2020_paper.html

        Args:
            smap (Tensor): the saliency map to calc weight.
            gmap (Tensor): the group mapping (group x channel)
            ctx (Context): the context of this process.

        Returns:
            Tensor: weights for each channel groups.
        """
        if DEBUG:
            len(ctx.activations) == 1
        activation: Tensor = ctx.activations[0]
        k: int = channel_shape(activation)
        u, v = position_shape(activation)
        g: int = batch_shape(gmap)
        if DEBUG:
            assert channel_shape(smap) == k
            assert position_shape(smap) == (u, v)
        ablation_list: List[Tensor] = list()
        for i in range(g):
            # group mask
            gmask: Tensor = torch.where(gmap[i, :].view(-1) > 0)[0]
            # drop i-th group (ablation)
            ablation: Tensor = activation.clone()
            ablation[:, gmask, :, :] = 0
            ablation_list.append(ablation)
        ablations: Tensor = torch.cat(ablation_list, dim=0)
        # forward Ablations and retrieve Ablation score
        a_scores: Tensor = ctx.classify_fn(activation=ablations)[
            :, ctx.label
        ].view(-1)
        # weight = slope of Ablation score
        return (ctx.score - a_scores) / (ctx.score + self.eps)

    def _channel_abscission_weight(
        self: ChannelSMAPS,
        smap: Tensor,
        gmap: Tensor,
        ctx: Context,
    ) -> Tensor:
        """Score-CAM

        H. Wang, et al.
        "Score-CAM:
        Score-Weighted Visual Explanations for Convolutional Neural Networks"
        CVF 2020.

        https://arxiv.org/abs/1910.01279

        calc CIC (Channel-wise Increase of Confidence) scores
        as a weight for each channels.
        Here, I named this method as "Abscission"
        (in contrast with "Ablation").

        Args:
            smap (Tensor): the saliency map to calc weight.
            gmap (Tensor): the group mapping (group x channel)
            ctx (Context): the context of this process.

        Returns:
            Tensor: weights for each channel groups.
        """
        # create grouped saliency map
        b, k, u, v = smap.shape
        g: int = batch_shape(gmap)
        group_smap: Tensor = (gmap @ smap.view(k, -1)).view(1, g, u, v)
        mask_dicts: List[Dict] = list()
        for i in range(g):
            # extract i-th group from the saliency map ("Abscission")
            abscission: Tensor = group_smap.clone()[:, [i], :, :]
            # normalize
            smax: Tensor = abscission.max().squeeze()
            smin: Tensor = abscission.min().squeeze()
            sdif: Tensor = smax - smin
            if sdif < self.eps:
                continue
            # create smoother mask for the original image
            mask: Tensor = (abscission - smin) / sdif
            # stack information
            mask_dicts.append(
                {
                    "group": i,
                    "key": sdif.detach().cpu().numpy().ravel()[0],
                    "mask": mask,
                }
            )
        if self.n_channels_ > 0:
            mask_dicts = sorted(
                mask_dicts, key=lambda x: x["key"], reverse=True
            )[: self.n_channels_]
        # create masked image
        masked_list: List[Tensor] = list()
        for mask_dict in mask_dicts:
            masked: Tensor = ctx.image * mask_dict["mask"]
            # SIGCAM
            # Q. Zhang, et al.
            # "A Novel Visual Interpretability for Deep Neural Networks
            # by Optimizing Activation Maps with Perturbation"
            # AAAI 2021.
            masked += ctx.blurred_image * (1.0 - mask_dict["mask"])
            masked_list.append(masked)
        maskedes: Tensor = torch.cat(masked_list, dim=0)
        # forward network, calc CIC scores and normalize it
        # (CIC: Channel-wise Increase of Confidence)
        # CIC score = (Abscission masked score) - (original score)
        # ## normalize CIC scores
        # The paper of Score-CAM use softmax for normalization.
        # But softmax turns negative values into positive values.
        # For that reason, if the target class is not the first rank,
        # the positive region of saliency map can't show the target.
        # So, Here, use normalization with L2-norm instead of softmax.
        cic_scores: Tensor = F.normalize(
            ctx.forward_fn(image=maskedes)[:, ctx.label].squeeze() - ctx.score,
            dim=0,
        )
        del maskedes
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

    def create_channel_smaps(
        self: ChannelSMAPS,
        position_smaps: SaliencyMaps,
        ctx: Context,
    ) -> SaliencyMaps:
        """merge channel saliency maps over channels.
        (merged saliency maps = layer saliency maps)

        Args:
            position_smaps (SaliencyMaps): position saliency maps.
            ctx (Context): the context of this process.

        Returns:
            SaliencyMaps: channel saliency maps.
        """
        # enlarge position saliency maps if channel_weight == "abscission"
        smaps: SaliencyMaps
        if self.channel_weight == "abscission":
            smaps = ctx.enlarge_fn(smaps=position_smaps)
        else:
            smaps = position_smaps
        # create channel saliency map for each layers
        channel_smaps: SaliencyMaps = SaliencyMaps()
        for smap in smaps:
            b, k, u, v = smap.size()
            if DEBUG:
                assert b == 1
            if k == 1:
                # channel saliency map is already created
                channel_smaps.append(smap.clone())
                continue
            # functions to create group mapping
            group_fn: Callable[[Tensor, Context], Tensor]
            if self.channel_group == "none":
                group_fn = self._channel_none_group
            elif self.channel_group == "k-means":
                group_fn = self._channel_kmeans_group
            elif self.channel_group == "spectral":
                group_fn = self._channel_spectral_group
            else:
                raise SystemError(
                    f"invalid channel_group: {self.channel_group}"
                )
            # group mapping
            gmap: Tensor = group_fn(smap=smap, ctx=ctx)
            if DEBUG:
                assert channel_shape(gmap) == k
                assert torch.allclose(
                    gmap.sum(dim=1), torch.ones(batch_shape(gmap))
                )
            # functions to create weight for each channel groups
            weight_fn: Callable[[Tensor, Tensor, Context], Tensor]
            if self.channel_weight == "none":
                weight_fn = self._channel_none_weight
            elif self.channel_weight == "eigen":
                weight_fn = self._channel_eigen_weight
            elif self.channel_weight == "ablation":
                weight_fn = self._channel_ablation_weight
            elif self.channel_weight == "abscission":
                weight_fn = self._channel_abscission_weight
            else:
                raise SystemError(
                    f"invalid channel_weight: {self.channel_weight}"
                )
            # average position saliency maps per channel groups
            g: int = batch_shape(gmap)
            group_smap: Tensor = (gmap @ smap.view(k, -1)).view(1, g, u, v)
            # weight group saliency map and sum over channels
            weight: Tensor = weight_fn(smap=smap, gmap=gmap, ctx=ctx)
            if self.channel_minmax_:
                # cognition-base and cognition-scissors
                base: int = weight.argmax().detach().cpu().numpy().ravel()[0]
                scis: int = weight.argmin().detach().cpu().numpy().ravel()[0]
                weight = torch.zeros(g).to(self.device)
                weight[base] = 1.0
                weight[scis] = -1.0
            weight = weight.view(1, g, 1, 1)
            channel_smaps.append(
                smap=(weight * group_smap).sum(dim=1, keepdim=True)
            )
        # finelize
        channel_smaps.finalize()
        smaps.clear()
        return channel_smaps
