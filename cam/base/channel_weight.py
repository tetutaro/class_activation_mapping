#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Optional, Callable, Any
from warnings import catch_warnings, filterwarnings

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
    CommonWeight,
)
from cam.base.containers import SaliencyMaps
from cam.base.context import Context


class ChannelWeight(CommonWeight):
    """A part of the CAM model that is responsible for channel saliency maps.

    XXX

    Args:
        channel_weight (str): the method of weighting for each channels.
        channel_group (str): the method of creating groups.
        channel_cosine (bool): if True, use cosine distance at clustering.
        channel_minmax (bool): if True, adopt the best&worst channel only.
        normalize_softmax (bool): normalize abscission score using softmax.
        n_channels (int): the number of abscission channel groups to calc.
        n_groups (Optional[int]): the number of channel groups.
    """

    def __init__(
        self: ChannelWeight,
        channel_weight: str,
        channel_group: str,
        channel_cosine: bool,
        channel_minmax: bool,
        normalize_softmax: bool,
        n_channels: int,
        n_groups: Optional[int],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # store flags
        self.channel_weight: str = channel_weight
        self.channel_group: str = channel_group
        self.channel_cosine_: bool = channel_cosine
        self.channel_minmax_: bool = channel_minmax
        self.normalize_softmax_: bool = normalize_softmax
        self.n_channels_: int = n_channels
        self.n_groups_: Optional[int] = n_groups
        if self.channel_weight == "none":
            self.channel_minmax_ = False
        self.inertias_: Dict[str, List[float]]
        return

    # ## functions to create channel group map (channel -> channel group)

    def _create_features(
        self: ChannelWeight,
        smap: Tensor,
    ) -> np.ndarray:
        """create feature matrix to clustering channels.

        Args:
            smap (Tensor): the saliemcy map (1 x channel x height x width).

        Returns:
            np.ndarray:
                feature matrix (channel x position).
                (position = height * width)
        """
        k: int = channel_shape(smap)
        # create channel x position matrix
        CP: Tensor = smap.view(k, -1)
        if not self.channel_cosine_:
            return CP.detach().cpu().numpy()
        CP_std: Tensor = (CP - CP.mean()) / CP.std()
        # SVD (singular value decomposition)
        # Cs = channel space
        Cs, _, _ = torch.linalg.svd(CP_std, full_matrices=False)
        return F.normalize(Cs.real, dim=1).detach().cpu().numpy()

    def _estimate_n_groups(
        self: ChannelWeight,
        features: np.ndarray,
    ) -> int:
        """automatically estimate an optimal number of groups.

        Args:
            features (np.ndarray): the data (n_data x n_features).

        Returns:
            int: the optimal number of groups
        """
        # estimate the max number of groups based on the Sturges' Rule
        n_data: int = features.shape[0]
        max_groups: int = int(np.ceil(np.log2(n_data))) + 2
        if max_groups < 5:
            return 3
        n_groups_list: List[int] = list(range(3, max_groups))
        # k-Means for earch number of groups
        inertia_list: List[float] = list()
        for n_groups in n_groups_list:
            km: KMeans = KMeans(
                n_clusters=n_groups,
                init="k-means++",
                n_init="auto",
                random_state=self.random_state,
            ).fit(features)
            inertia_list.append(km.inertia_)
        # calc elbow point of k-Means' inertia
        with catch_warnings():
            filterwarnings("ignore", category=UserWarning)
            kneedle: KneeLocator = KneeLocator(
                x=n_groups_list,
                y=inertia_list,
                curve="convex",
                direction="decreasing",
            )
        self.inertias_ = {
            "n_clusters": [float(x) for x in n_groups_list],
            "inertia": inertia_list,
        }
        n_groups: int
        if kneedle.elbow is None:
            n_groups = max_groups
        else:
            n_groups = kneedle.elbow
        return n_groups

    def _channel_group_none(
        self: ChannelWeight,
        smap: Tensor,
        ctx: Context,
    ) -> Tensor:
        """dummy clustering.

        Args:
            smap (Tensor): the saliemcy map (1 x channel x height x width).
            ctx (Context): the context of this process.

        Returns:
            Tensor: identity matrix (channel x channel).
        """
        return torch.diag(torch.ones(channel_shape(smap))).to(self.device)

    def _channel_group_kmeans(
        self: ChannelWeight,
        smap: Tensor,
        ctx: Context,
    ) -> Tensor:
        """clustering channels using k-Means.

        Args:
            smap (Tensor): the saliemcy map (1 x channel x height x width).
            ctx (Context): the context of this process.

        Returns:
            Tensor: channel group map (group x channel).
        """
        k: int = channel_shape(smap)
        # create features to cluster channels
        features: np.ndarray = self._create_features(smap=smap)
        if self.n_groups_ is None:
            # estimate the optimal number of groups
            self.n_groups_ = self._estimate_n_groups(features=features)
        # cluster channels using k-Means
        labels: np.ndarray = (
            KMeans(
                n_clusters=self.n_groups_,
                init="k-means++",
                n_init="auto",
                random_state=self.random_state,
            )
            .fit_predict(features)
            .reshape(-1)
        )
        # create group map
        group_weight_list: List[List[float]] = list()
        for group in range(self.n_groups_):
            group_weight: np.ndarray = np.where(labels == group, 1.0, 0.0)
            group_weight /= group_weight.sum()
            group_weight_list.append(group_weight.tolist())
        gmap: Tensor = torch.tensor(group_weight_list).to(self.device)
        if DEBUG:
            assert channel_shape(gmap) == k
            assert torch.allclose(
                gmap.sum(dim=1), torch.ones(batch_shape(gmap))
            )
        return gmap

    def _channel_group_spectral(
        self: ChannelWeight,
        smap: Tensor,
        ctx: Context,
    ) -> Tensor:
        """clustering channels using Spectral Clustering.

        Args:
            smap (Tensor): the saliemcy map (1 x channel x height x width).
            ctx (Context): the context of this process.

        Returns:
            Tensor: channel group map (group x channel).
        """
        k: int = channel_shape(smap)
        # create features to cluster channels
        features: np.ndarray = self._create_features(smap=smap)
        if self.n_groups_ is None:
            # estimate the optimal number of groups
            self.n_groups_ = self._estimate_n_groups(features=features)
        # cluster channels using Spectral Clustering
        with catch_warnings():
            filterwarnings("ignore", category=UserWarning)
            labels: np.ndarray = (
                SpectralClustering(
                    n_clusters=self.n_groups_,
                    affinity="nearest_neighbors",
                    n_jobs=-1,
                    random_state=self.random_state,
                )
                .fit_predict(features)
                .reshape(-1)
            )
        # create group map
        group_weight_list: List[List[float]] = list()
        for group in range(self.n_groups_):
            group_weight: np.ndarray = np.where(labels == group, 1.0, 0.0)
            group_weight /= group_weight.sum()
            group_weight_list.append(group_weight.tolist())
        gmap: Tensor = torch.tensor(group_weight_list).to(self.device)
        if DEBUG:
            assert channel_shape(gmap) == k
            assert torch.allclose(
                gmap.sum(dim=1), torch.ones(batch_shape(gmap))
            )
        return gmap

    # ## functions to create weight for each channel group

    def _channel_weight_none(
        self: ChannelWeight,
        g_smap: Tensor,
        gmap: Tensor,
        ctx: Context,
    ) -> Tensor:
        """weight by the number of channels for each channel groups.

        Args:
            g_smap (Tensor):
                the grouped saliency map (1 x group x height x width).
            gmap (Tensor): the channel group map (group x channel).
            ctx (Context): the context of this process.

        Returns:
            Tensor: weight for each channel groups.
        """
        return torch.where(gmap > 0, 1.0, 0.0).sum(dim=1) / channel_shape(gmap)

    def _channel_weight_eigen(
        self: ChannelWeight,
        g_smap: Tensor,
        gmap: Tensor,
        ctx: Context,
    ) -> Tensor:
        """the first eigen-vector of the channel space.

        The grouped saliency map is divided into
        the channel group space and the positoin space by
        SVD (Singular Value Decomposition).
        Each vertical vectors in the channel group space
        is the eigen-vector of the channel group space.
        use the first eigen-vector (that has the highest eigen-value)
        of the channel group space as the weight for each channel groups.

        Args:
            g_smap (Tensor):
                the grouped saliency map (1 x group x height x width).
            gmap (Tensor): the channel group map (group x channel).
            ctx (Context): the context of this process.

        Returns:
            Tensor: weight for each channel groups.
        """
        g: int = channel_shape(g_smap)
        # create group x position matrix
        GP: Tensor = g_smap.view(g, -1)
        # standarize
        GP_std: Tensor = (GP - GP.mean()) / GP.std()
        # SVD (singular value decomposition)
        # Gs = channel group space, ss = eigen-values
        Gs, ss, _ = torch.linalg.svd(GP_std, full_matrices=False)
        # retrieve the first eigen-vector and normalize it
        weight: Tensor = F.normalize(Gs.real[:, ss.argmax()], dim=0)
        # check the sign of weight
        if (
            weight.view(1, g, 1, 1) * F.relu(g_smap - g_smap.median())
        ).sum() < 0:
            weight = -weight
        return weight

    def _channel_weight_ablation(
        self: ChannelWeight,
        g_smap: Tensor,
        gmap: Tensor,
        ctx: Context,
    ) -> Tensor:
        """slope of Ablation score.

        Args:
            g_smap (Tensor):
                the grouped saliency map (1 x group x height x width).
            gmap (Tensor): the group map (group x channel).
            ctx (Context): the context of this process.

        Returns:
            Tensor: weight for each channel groups.
        """
        if DEBUG:
            len(ctx.activations) == 1
        activation: Tensor = ctx.activations[0]
        u, v = position_shape(activation)
        g: int = batch_shape(gmap)
        ablation_list: List[Tensor] = list()
        for i in range(g):
            # group mask
            gmask: Tensor = torch.where(gmap[i, :].view(-1) > 0)[0]
            # drop i-th channel group (ablation)
            ablation: Tensor = activation.clone()
            ablation[:, gmask, :, :] = 0
            ablation_list.append(ablation)
        ablations: Tensor = torch.cat(ablation_list, dim=0)
        # forward Ablations and retrieve Ablation score
        a_scores: Tensor = ctx.classify_fn(activation=ablations)[:, ctx.label]
        # slope of Ablation score
        return (ctx.score - a_scores) / (ctx.score + self.eps)

    def _channel_weight_abscission(
        self: ChannelWeight,
        g_smap: Tensor,
        gmap: Tensor,
        ctx: Context,
    ) -> Tensor:
        """CIC (Channel-wise Increase of Confidence) scores.

        calc CIC (Channel-wise Increase of Confidence) scores
        as a weight for each channels.
        Here, I named this method as "Abscission"
        (in contrast with "Ablation").

        Args:
            g_smap (Tensor):
                the grouped saliency map (1 x group x height x width).
            gmap (Tensor): the group map (group x channel).
            ctx (Context): the context of this process.

        Returns:
            Tensor: weight for each channel groups.
        """
        # create grouped saliency map
        _, g, u, v = g_smap.shape
        mask_dicts: List[Dict] = list()
        for i in range(g):
            # extract i-th group from the saliency map ("Abscission")
            abscission: Tensor = g_smap[:, [i], :, :]
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
        if self.n_channels_ > 0:
            mask_dicts = sorted(
                mask_dicts, key=lambda x: x["key"], reverse=True
            )[: self.n_channels_]
        assert len(mask_dicts) > 1
        # get baseline score and logit
        base_score: Tensor = F.sigmoid(
            ctx.forward_fn(image=ctx.blurred_image)
        )[:, ctx.label].squeeze()
        base_logit: Tensor = (base_score / (1.0 - base_score)).log()
        # create masked image
        masked_list: List[Tensor] = list()
        for mask_dict in mask_dicts:
            masked: Tensor = ctx.image * mask_dict["mask"]
            # SIGCAM
            masked += ctx.blurred_image * (1.0 - mask_dict["mask"])
            masked_list.append(masked)
        maskedes: Tensor = torch.cat(masked_list, dim=0)
        # forward network then retrieve abscission score and logit
        absc_score: Tensor = F.sigmoid(ctx.forward_fn(image=maskedes))[
            :, ctx.label
        ]
        absc_logit: Tensor = (absc_score / (1.0 - absc_score)).log()
        del maskedes
        # calc CIC score and normalize
        cic_scores: Tensor
        if self.normalize_softmax_:
            cic_scores = F.softmax(absc_logit - base_logit, dim=0)
        else:
            cic_scores = F.normalize(absc_logit - base_logit, dim=0)
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

    def weight_channel(
        self: ChannelWeight,
        smaps: SaliencyMaps,
        ctx: Context,
    ) -> SaliencyMaps:
        """group and weight each channels and sum them up.

        Args:
            smaps (SaliencyMaps): saliency maps.
            ctx (Context): the context of this process.

        Returns:
            SaliencyMaps: summed saliency maps.
        """
        if self.channel_weight == "abscission":
            # forcibly enlarge saliency maps to the original image size
            smaps = ctx.enlarge_fn(smaps=smaps)
        # weight and merge channels for each layers
        merged_smaps: SaliencyMaps = SaliencyMaps()
        for smap in smaps:
            _, k, u, v = smap.size()
            if k == 1:
                # channel saliency map is already created
                merged_smaps.append(smap.clone())
                continue
            # the function to create channel group map
            group_fn: Callable[[Tensor, Context], Tensor]
            if self.channel_group == "none":
                group_fn = self._channel_group_none
            elif self.channel_group == "k-means":
                group_fn = self._channel_group_kmeans
            elif self.channel_group == "spectral":
                group_fn = self._channel_group_spectral
            else:
                raise SystemError(
                    f"invalid channel_group: {self.channel_group}"
                )
            # the function to create weight for each channel groups
            weight_fn: Callable[[Tensor, Tensor, Context], Tensor]
            if self.channel_weight == "none":
                weight_fn = self._channel_weight_none
            elif self.channel_weight == "eigen":
                weight_fn = self._channel_weight_eigen
            elif self.channel_weight == "ablation":
                weight_fn = self._channel_weight_ablation
            elif self.channel_weight == "abscission":
                weight_fn = self._channel_weight_abscission
            else:
                raise SystemError(
                    f"invalid channel_weight: {self.channel_weight}"
                )
            # create channel group map
            gmap: Tensor = group_fn(smap=smap, ctx=ctx)
            # group saliency map
            g: int = batch_shape(gmap)
            g_smap: Tensor = (gmap @ smap.view(k, -1)).view(1, g, u, v)
            # weight grouped saliency map and sum over channels
            g_weight: Tensor = weight_fn(g_smap=g_smap, gmap=gmap, ctx=ctx)
            if self.channel_minmax_:
                # cognition-base and cognition-scissors
                # cognition-base is the channel group that has
                # the highest impact on the saliency map.
                # and cognision-scissors is the channel group that has
                # the least impact on the saliency map.
                # set the value of cognition-base of the weight to 1,
                # and set the value of cognition-scissors of the weight to -1.
                # other values of the weight is set to 0.
                base: int = g_weight.argmax().detach().cpu().numpy().ravel()[0]
                scis: int = g_weight.argmin().detach().cpu().numpy().ravel()[0]
                val: float = float(g_weight.detach().cpu().numpy()[base])
                g_weight = torch.zeros(g).to(self.device)
                g_weight[base] = val
                g_weight[scis] = -(1.0 - val)
            g_weight = g_weight.view(1, g, 1, 1)
            # merge channels (weighted sum)
            g_smap = (g_weight * g_smap).sum(dim=1, keepdim=True)
            # centerize
            g_smap -= g_smap.median()
            merged_smaps.append(smap=g_smap)
        # finelize
        merged_smaps.finalize()
        smaps.clear()
        return merged_smaps
