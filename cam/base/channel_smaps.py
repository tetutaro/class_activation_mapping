#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Optional, Callable, Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from cam.base import DEBUG, batch_shape, channel_shape, CommonSMAP
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
        return

    def _channel_mean_weight(
        self: ChannelSMAPS,
        smap: Tensor,
        ctx: Context,
    ) -> Tensor:
        """global average pooling over channels.

        Args:
            smap (Tensor): the saliency map to calc weight.
            ctx (Context): the context of this process.

        Returns:
            Tensor: the weight for each channels.
        """
        k: int = channel_shape(smap)
        return torch.tensor([1 / k] * k).view(1, k, 1, 1)

    def _channel_eigen_weight(
        self: ChannelSMAPS,
        smap: Tensor,
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
            ctx (Context): the context of this process.

        Returns:
            Tensor: the weight for each channels.
        """
        k: int = channel_shape(smap)
        # create channel (k) x position (u x v) matrix
        CP: Tensor = smap.view(k, -1)
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
        if self.channel_weight == "cosine":
            raise NotImplementedError("cosine is not implemented yet")
        weight: Tensor = Ch.real[:, SS.argmax()]
        # normalize weight
        weight = F.normalize(weight, dim=0).view(1, k, 1, 1)
        # the eigen-vector may have the opposite sign
        if (weight * F.relu(smap)).sum() < 0:
            weight = -weight
        return weight

    def _channel_ablation_weight(
        self: ChannelSMAPS,
        smap: Tensor,
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
            ctx (Context): the context of this process.

        Returns:
            Tensor: the weight for each channels.
        """
        k: int = channel_shape(smap)
        ablated_list: List[Tensor] = list()
        for i in range(k):
            # drop i-th channel (ablation)
            ablated_map: Tensor = smap.clone()
            ablated_map[:, i, :, :] = 0.0
            ablated_list.append(ablated_map)
        ablateds: Tensor = torch.cat(ablated_list, dim=0)
        scores: Tensor = ctx.classify_fn(activations=ablateds, label=ctx.label)
        weight: Tensor = (ctx.score - scores) / (ctx.score + self.eps)
        return weight.view(1, k, 1, 1)

    def _channel_abscission_weight(
        self: ChannelSMAPS,
        smap: Tensor,
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
            ctx (Context): the context of this process.

        Returns:
            Tensor: the weight for each channels.
        """
        k: int = channel_shape(smap)
        mask_dicts: List[Dict] = list()
        for i in range(k):
            # extract i-th channel from the saliency map ("Abscission")
            absc: Tensor = smap.clone()[:, [i], :, :]
            # normalize
            smax: float = absc.detach().max().cpu().numpy().ravel()[0]
            smin: float = absc.detach().min().cpu().numpy().ravel()[0]
            sdif: float = smax - smin
            if sdif < self.eps:
                continue
            # create smoother mask of the original image
            mask: Tensor = (absc - smin) / sdif
            # stack information
            mask_dicts.append(
                {
                    "channel": i,
                    "key": sdif,
                    "mask": mask,
                }
            )
        if self.n_channels_ > 0:
            mask_dicts = sorted(
                mask_dicts, key=lambda x: x["key"], reverse=True
            )[: self.n_channels_]
        # calc CIC scores
        masked_list: List[Tensor] = list()
        for mask_dict in mask_dicts:
            # create masked image
            masked: Tensor = ctx.image * mask_dict["mask"]
            # SIGCAM
            # Q. Zhang, et al.
            # "A Novel Visual Interpretability for Deep Neural Networks
            # by Optimizing Activation Maps with Perturbation"
            # AAAI 2021.
            # masked += ctx.blurred_image * (1.0 - mask_dict["mask"])
            masked_list.append(masked)
        masked_images: Tensor = torch.cat(masked_list, dim=0)
        # forward network
        masked_scores: np.ndarray = (
            ctx.forward_fn(image=masked_images)[:, ctx.label]
            .detach()
            .squeeze()
            .cpu()
            .numpy()
            .ravel()
        )
        # CIC scores
        cic_scores: List[float] = (masked_scores - ctx.score).tolist()
        # normalize CIC scores
        # The paper of Score-CAM use softmax for normalization.
        # But softmax turns negative values into positive values.
        # For that reason, if the target class is not the first rank,
        # the positive region of saliency map can't show the target.
        # So, Here, use normalization with L2-norm instead of softmax.
        normed_cic_scores: Tensor = F.normalize(
            torch.tensor(cic_scores).to(self.device), dim=0
        )
        # create weight
        weight: Tensor
        if batch_shape(normed_cic_scores) < k:
            idx: Tensor = torch.tensor([x["channel"] for x in mask_dicts]).to(
                self.device
            )
            weight = (
                torch.zeros(k)
                .scatter(dim=0, index=idx, src=normed_cic_scores)
                .to(self.device)
            )
            del cic_scores
        else:
            weight = cic_scores
        return weight.view(1, k, 1, 1)

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
        # create the weight for each channel
        # and create channel saliency map for each layers
        channel_smaps: SaliencyMaps = SaliencyMaps()
        for smap in smaps:
            if DEBUG:
                assert batch_shape(smap) == 1
            k: int = channel_shape(smap)
            if k == 1:
                # saliency map is already created
                channel_smaps.append(smap.clone())
                continue
            fn: Callable[[Tensor, Context], Tensor]
            if self.channel_weight == "none":
                fn = self._channel_mean_weight
            elif self.channel_weight in ["eigen", "cosine"]:
                fn = self._channel_eigen_weight
            elif self.channel_weight == "ablation":
                fn = self._channel_ablation_weight
            elif self.channel_weight == "abscission":
                fn = self._channel_abscission_weight
            else:
                raise SystemError(
                    f"invalid channel_weight: {self.channel_weight}"
                )
            # create the weight and multiply it to position saliency map
            channel_smaps.append(
                smap=(fn(smap=smap, ctx=ctx) * smap).sum(dim=1, keepdim=True)
            )
        channel_smaps.finalize()
        smaps.clear()
        return channel_smaps
