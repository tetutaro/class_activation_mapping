#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from PIL import Image, ImageFilter
import matplotlib as mpl

from cam.cnn import CNN
from cam.utils import draw_image_heatmap
from cam.libs_cam import TargetLayer, ResourceCNN, Weights, SaliencyMaps
from cam.libs_cam import batch_size, channel_size


class BaseCAM(CNN, ABC):
    """The common class for all CAM models.

    The aim of CAM model is to create a "Saliency Map" that represents
    which regions of the original image
    the CNN model pays attention (or doesn't pay attention)
    to determine the indicated label.

    There are following 5 processes to create the saliency map.

    #. forward the image to CNN and get activations
        * if needed, backward the score and get gradients (require_grad=True)
        * self._forward()
    #. create weights for activations
        * self._create_weights()
    #. weight activation with the weights and enlarge them to the image size
        * I named the weighted activation as "Channel Saliency Map"
        * self._create_channel_smaps()
    #. merge channel saliency maps over channels
        * I named the merged channel saliency map as "Layer Saliency Map"
        * self._create_layer_smaps()
    #. merge enlarged layer saliency maps over layers
        * and then, the final saliency map is created
        * self._merge_layer_smaps()

    The final saliency map also be able to call "Heatmap"
    that represents which regions the CNN model pays attention (or doesn't)
    with colors.

    Here, the aim of CAM models is shown by displaying the heatmap
    overlaid on the original image. (self._draw_image())

    Only you have to do is create instance of the class of CAM model
    and call model.draw().

    Args:
        resource (ResourceCNN): resouce of the CNN model.
        target (TargetLayer): target Conv. Layers to retrieve activations.
        requires_grad (bool):
            if True, do backward and retrive gradient(s).
        blur_image (bool):
            create a blurred image to stabilize the saliency map.
        channel_weight (Optional[str]):
            how to create weight for each channels.
    """

    def __init__(
        self: BaseCAM,
        resource: ResourceCNN,
        target: TargetLayer,
        requires_grad: bool = False,
        blur_image: bool = False,
        channel_weight: Optional[str] = None,
    ) -> None:
        super().__init__(target=target, **resource)
        # store flags
        self.requires_grad_: bool = requires_grad
        self.blur_image_: bool = blur_image
        self.channel_weight_: Optional[str] = channel_weight
        # caches created in self._forward()
        self.raw_image_: Optional[Image] = None
        self.image_: Optional[Tensor] = None
        self.blurred_image_: Optional[Tensor] = None
        self.width_: Optional[int] = None
        self.height_: Optional[int] = None
        self.target_class_: Optional[int] = None
        self.score_: Optional[float] = None
        # constants
        self.eps_ = 1e-6
        # the name of CAM model
        self.name_: str
        self._set_name()
        return

    @abstractmethod
    def _set_name(self: BaseCAM) -> None:
        """set the name of CAM model for the title of the output image."""
        self.name_ = ""
        return

    def _create_image(
        self: BaseCAM,
        path: str,
    ) -> None:
        """load image and create tensors.

        Args:
            path (str): the pathname of the original image.
        """
        # load the original image
        self.raw_image_ = Image.open(path)
        self.width_, self.height_ = self.raw_image_.size
        self.image_ = (
            self.transform_(self.raw_image_).unsqueeze(0).to(self.device_)
        )
        # create the blurred image
        if self.blur_image_:
            self.blurred_image_: Tensor = (
                self.transform_(
                    self.raw_image_.filter(
                        filter=ImageFilter.GaussianBlur(radius=51)
                    )
                )
                .unsqueeze(0)
                .to(self.device)
            )
        return

    def _forward(
        self: BaseCAM,
        image: Optional[Tensor] = None,
        requires_grad: Optional[bool] = None,
        rank: Optional[int] = None,
        label: Optional[int] = None,
    ) -> None:
        """input the image to the CNN.

        If both rank and label are specified, label takes precedence.
        If both rank and label are not specified, set rank to 0.

        Args:
            image (Optional[Tensor]): the image to forward the network.
            requires_grad (Optional[bool]): gradients are needed.
            rank (Optional[int]): the rank of the target class.
            label (Optional[int]): the label of the target class.
        """
        # check indicated image and requires_grad
        if image is None:
            image = self.image_
        if requires_grad is None:
            requires_grad = self.requires_grad_
        # check indicated rank and label
        if self.target_class_ is None:
            if rank is None and label is None:
                rank = 0
            if (rank is not None and rank >= 1000) or (
                label is not None and label >= 1000
            ):
                raise ValueError("rank and label must be less than 1000")
        # clear activations and gradients
        self.activations_.clear()
        self.gradients_.clear()
        # forward the image to the network
        if requires_grad:
            scores: Tensor = self._forward_net(image=self.image_)
        else:
            with torch.no_grad():
                scores: Tensor = self._forward_net(image=self.image_)
        # decide the target class
        if self.target_class_ is None:
            if label is not None:
                self.target_class_ = label
            else:
                self.target_class_ = (
                    scores.clone()
                    .detach()
                    .argsort(dim=1, descending=True)
                    .squeeze()
                    .cpu()
                    .numpy()[rank]
                )
        # get the score of target class
        self.score_ = (
            scores.clone().detach().squeeze().cpu().numpy()[self.target_class_]
        )
        # backward the score to the network
        if requires_grad:
            self.net_.zero_grad()
            scores[:, [self.target_class_]].backward(retain_graph=None)
        # finalize activations and gradients
        self.activations_.finalize()
        self.gradients_.finalize()
        return

    @abstractmethod
    def _create_weights(self: BaseCAM, **kwargs: Any) -> Weights:
        """create the weight.

        Returns:
            Weights: the weight.
        """
        weights: Weights = Weights()
        weights.finalize()
        return weights

    def _dummy_weights(self: BaseCAM) -> Weights:
        """the dummy weights (each value of weight = 1).

        Returns:
            Weights: the dummy weight.
        """
        weights: Weights = Weights()
        for _, (_, k, _, _) in self.activations_:
            weights.append(torch.ones((1, k, 1, 1)).to(self.device_))
        weights.finalize()
        return weights

    def _class_weights(self: BaseCAM) -> Weights:
        """the weights of the target class from self.class_weights_.

        If this function is used, the target of Conv. Layer must be the "last".

        Returns:
            Weights: the weight of the target class.
        """
        weights: Weights = Weights()
        weights.append(
            self.class_weights_[[self.target_class_], :].view(
                1, channel_size(self.class_weights_), 1, 1
            )
        )
        weights.finalize()
        return weights

    def _grad_weights(self: BaseCAM) -> Weights:
        """create the weights from gradients.

        weight = average gradient per channel over position.

        Returns:
            Weights: the weight
        """
        weights: Weights = Weights()
        for gradient, (_, k, _, _) in self.gradients_:
            weights.append(
                gradient.view(1, k, -1).mean(dim=2).view(1, k, 1, 1)
            )
        weights.finalize()
        return weights

    def _grad_pp_weights(self: BaseCAM) -> Weights:
        """create the weights from gradients.

        the weights are calced according to formulae int the Grad-CAM++ paper.

        Returns:
            Weights: the weight
        """
        weights: Weights = Weights()
        for (activation, (_, k, _, _)), (gradient, _) in zip(
            self.activations_, self.gradients_
        ):
            # calc alpha (the eq (19) in the paper of Grad-CAM++)
            alpha_numer: Tensor = gradient.pow(2.0)
            alpha_denom: Tensor = 2.0 * alpha_numer
            alpha_denom += (
                (gradient.pow(3.0) * activation)
                .view(1, k, -1)
                .sum(dim=2)
                .view(1, k, 1, 1)
            )
            alpha_denom = (
                torch.where(
                    alpha_denom != 0.0,
                    alpha_denom,
                    torch.ones_like(alpha_denom),
                )
                + self.eps_
            )  # for stability
            alpha: Tensor = alpha_numer / alpha_denom
            weights.append(
                (np.exp(self.score_) * alpha * gradient)
                .view(1, k, -1)
                .sum(dim=2)
                .view(1, k, 1, 1)
            )
        weights.finalize()
        return weights

    def _create_channel_smaps(
        self: BaseCAM,
        weights: Weights,
    ) -> SaliencyMaps:
        """multiply activations with weights and enlarge them to the image size.
        (weighted activations = channel saliency maps)

        Args:
            weights (Weights): weights

        Returns:
            SaliencyMaps: channel saliency maps
        """
        channel_saliencies: SaliencyMaps = SaliencyMaps()
        assert len(weights) == len(self.activations_)
        for (weight, _), (activation, _) in zip(weights, self.activations_):
            channel_saliencies.append(
                F.interpolate(
                    weight * activation,
                    size=(self.height_, self.width_),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        channel_saliencies.finalize()
        return channel_saliencies

    def _calc_eigen_weight(
        self: BaseCAM,
        salient: Tensor,
    ) -> Tensor:
        """calc a weight for each channels using SVD
        (Singular Value Decomposition).

        Args:
            salient (Tensor): the saliency map to calc weight.

        Returns:
            Tensor: the weight for each channels.
        """
        assert batch_size(salient) == 1
        k = channel_size(salient)
        if k == 1:
            return torch.tensor([1.0]).view(1, 1, 1, 1).to(self.device_)
        # create channel (k) x position (u x v) matrix
        CP: Tensor = salient.view(k, -1)
        # standarize
        CP_std: Tensor = (CP - CP.mean()) / CP.std()
        # SVD (singular value decomposition)
        Ch, SS, _ = torch.linalg.svd(CP_std, full_matrices=False)
        # retrieve weight vector
        # * Candidate 1: (the same as the Eigen-CAM paper)
        #   the first eigen-vector of the channel matrix
        #     weight: Tensor = Ch.real[:, SS.argmax()]
        # * Candidate 2: L2-norm of each channel vector
        #     weight: Tensor = F.normalize(Ch.real, dim=0)
        # * Candidate 3: projection of each channel vector
        #   to the first eigen-vector of variance-covariance matrix
        #   of the channel matrix
        weight: Tensor = Ch.real[:, SS.argmax()]
        # normalize weight
        weight = F.normalize(weight, dim=0).view(1, k, 1, 1)
        # the eigen-vector may have the opposite sign
        if (weight * F.relu(salient)).sum() < 0:
            weight = -weight
        return weight

    def _calc_score_weight(
        self: BaseCAM,
        salient: Tensor,
        n_channels: Optional[int],
    ) -> Tensor:
        """calc CIC (Channel-wise Increase of Confidence) scores
        as a weight for each channels.

        "Abscission" (on the contract with "Ablation")

        Args:
            salient (Tensor): the saliency map to calc weight.
            n_channels (Optional[int]): the number of channels calc score.

        Returns:
            Tensor: the weight for each channels.
        """
        assert batch_size(salient) == 1
        k = channel_size(salient)
        if k == 1:
            return torch.tensor([1.0]).view(1, 1, 1, 1).to(self.device_)
        if n_channels is None:
            n_channels = -1
        smap_dicts: List[Dict] = list()
        for i in range(k):
            # extract i-th channel from the saliency map ("Abscission")
            smap: Tensor = salient.clone().detach()[:, [i], :, :]
            # normalize
            smax: float = smap.clone().detach().max().cpu().numpy().ravel()[0]
            smin: float = smap.clone().detach().min().cpu().numpy().ravel()[0]
            sdif: float = smax - smin
            if sdif < self.eps_:
                continue
            # create smoother mask of the original image
            # * Candidate 1: (the same as the original paper)
            #     mask: Tensor = (smap - smin) / sdif
            # * Candidate 2: (use positive value in smap)
            #     mask: Tensor = F.relu(smap) / smax
            # * Candidate 3: (allow negative mask)
            #     mask: Tensor = (smap / smax).clamp(min=-1.0, max=1.0)
            mask: Tensor = (smap - smin) / sdif
            # stack information
            smap_dicts.append(
                {
                    "channel": i,
                    "key": sdif,
                    "smap": smap,
                    "mask": mask,
                }
            )
        if n_channels > 0:
            smap_dicts = sorted(
                smap_dicts, key=lambda x: x["key"], reverse=True
            )[:n_channels]
        # calc CIC scores
        cic_scores: List[float] = list()
        for smap_dict in smap_dicts:
            # create masked image
            masked: Tensor = self.image_ * smap_dict["mask"]
            if self.blur_image_:
                # SIGCAM
                masked += self.blurred_image_ * (1.0 - smap_dict["mask"])
            # forward network
            with torch.no_grad():
                masked_scores: Tensor = self._forward_net(image=masked)
            masked_score: float = (
                masked_scores.detach()
                .squeeze()
                .cpu()
                .numpy()[self.target_class_]
            )
            self.activations_.clear()
            self.gradients_.clear()
            # calc CIC score
            # * Candidate 1: (the same as the original paper)
            #     cic_score: float = masked_score - self.score_
            # * Candidate 2: (instantaneous slope)
            #     cic_score: float = (
            #         (masked_score - self.score_) / (self_score_ + self.eps_)
            #     )
            cic_score: float = masked_score - self.score_
            cic_scores.append(cic_score)
        # normalize CIC scores
        # The paper of Score-CAM use softmax for normalization.
        # But softmax turns negative values into positive values.
        # For that reason, if the target class is not the first rank,
        # the positive region of saliency map can't show the target.
        # So, Here, use normalization with L2-norm instead of softmax.
        normed_cic_scores: Tensor = F.normalize(
            torch.tensor(cic_scores).to(self.device_), dim=0
        )
        # create weight
        weight: Tensor
        if n_channels > 0:
            idx: Tensor = torch.tensor([x["channel"] for x in smap_dicts]).to(
                self.device_
            )
            weight = (
                torch.zeros(k)
                .scatter(dim=0, index=idx, src=normed_cic_scores)
                .to(self.device_)
            )
        else:
            weight = normed_cic_scores
        return weight.view(1, k, 1, 1)

    def _create_layer_smaps(
        self: BaseCAM,
        channel_smaps: SaliencyMaps,
        channel_weight: Optional[str],
        n_channels: Optional[int],
    ) -> SaliencyMaps:
        """merge channel saliency maps over channels.
        (merged saliency maps = layer saliency maps)

        Args:
            channel_smaps (SaliencyMaps): channel saliency maps.
            channel_weight (Optional[str]):
                how to create weight for each channels.
            n_channels (Optional[int]): the number of channels calc score.

        Returns:
            SaliencyMaps: layer saliency maps.
        """
        if self.channel_weight_ is not None:
            channel_weight = self.channel_weight_
        layer_smaps: SaliencyMaps = SaliencyMaps(is_layer=True)
        for channel_smap, (b, k, _, _) in channel_smaps:
            assert b == 1
            if k == 1:
                # saliency map is already created
                layer_smaps.append(channel_smap)
                continue
            if channel_weight not in ["eigen", "score"]:
                # average over channels
                layer_smaps.append(
                    F.relu(channel_smap).mean(dim=1, keepdim=True)
                    - F.relu(-channel_smap).mean(dim=1, keepdim=True)
                )
                continue
            weight: Tensor
            if channel_weight == "eigen":
                # calc weight for each channels using SVD
                weight = self._calc_eigen_weight(
                    salient=channel_smap,
                )
            else:  # channel_weight == "score"
                # calc CIC scores as the weight for each channels
                weight = self._calc_score_weight(
                    salient=channel_smap,
                    n_channels=n_channels,
                )
            # weight channel saliency map
            channel_smap *= weight
            # sum over channels
            layer_smaps.append(
                F.relu(channel_smap).sum(dim=1, keepdim=True)
                - F.relu(-channel_smap).sum(dim=1, keepdim=True)
            )
        layer_smaps.finalize()
        return layer_smaps

    def _merge_layer_smaps(
        self: BaseCAM,
        layer_smaps: SaliencyMaps,
    ) -> np.ndarray:
        """merge layer saliency maps and conver it to heatmap.

        Args:
            saliency_maps (List[Tensor]): saliency_map(s).

        Returns:
            np.ndarray: heatmap.
        """
        assert len(layer_smaps) > 0
        stacked: Tensor = torch.stack([x[0] for x in layer_smaps], dim=0)
        return (
            (F.relu(stacked).sum(dim=0) - F.relu(-stacked).sum(dim=0))
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

    def _draw_image(
        self: BaseCAM,
        heatmap: np.ndarray,
        draw_negative: bool,
        fig: Optional[mpl.figure.Figure],
        ax: Optional[mpl.axes.Axes],
    ) -> None:
        """draw the image with the heatmap overlaied on the original image.

        Args:
            heatmap (np.ndarray): heatmap.
            fig (Optional[mpl.figure.Figure]):
                the Figure instance that the output image is drawn.
                if None, create it inside this function.
            ax (Optinonal[mpl.axies.Axes):
                the Axes instance that the output image is drawn.
                if None, create it inside this function.
        """
        # normalize heatmap
        heatmap = heatmap / heatmap.max()
        if draw_negative:
            heatmap = heatmap.clip(min=-1.0, max=1.0)
        else:
            heatmap = heatmap.clip(min=0.0, max=1.0)
        # title
        title: str = f"{self.name_}"
        if fig is None or ax is None:
            title += f" ({self.labels_[self.target_class_]})"
        # draw
        draw_image_heatmap(
            image=self.raw_image_,
            heatmap=heatmap,
            title=title,
            draw_negative=draw_negative,
            fig=fig,
            ax=ax,
        )
        return

    def _clear_cache(self: BaseCAM) -> None:
        """clear all caches."""
        super()._clear_cache()
        self.raw_image_ = None
        self.image_ = None
        self.blurred_image_ = None
        self.width_ = None
        self.height_ = None
        self.target_class_ = None
        self.score_ = None
        return

    def draw(
        self: BaseCAM,
        path: str,
        rank: Optional[int] = None,
        label: Optional[int] = None,
        channel_weight: Optional[str] = None,
        n_samples: Optional[int] = None,
        sigma: Optional[float] = None,
        random_state: Optional[int] = None,
        n_channels: Optional[int] = 10,
        draw_negative: bool = False,
        fig: Optional[mpl.figure.Figure] = None,
        ax: Optional[mpl.axes.Axes] = None,
    ) -> None:
        """the main function.

        Args:
            path (str): the pathname of the original image.
            rank (Optional[int]): the rank of the target class.
            label (Optional[int]): the label of the target class.
            channel_weight (bool):
                if True, weighted activation(s) are further weighted
                by the first eigenvector of decomposed matrix by SVD.
            n_samples (Optional[float]): number of samplings
            sigma (Optional[float]): standard deviation of noise
            random_state (Optional[int]): random seed
            n_channels (Optional[int]): the number of channels calc score.
            draw_negative (bool): draw negative regions.
            fig (Optional[mpl.figure.Figure]):
                the Figure instance that the output image is drawn.
            ax (Optinonal[mpl.axies.Axes):
                the Axes instance that the output image is drawn.
        """
        self._create_image(path=path)
        self._forward(rank=rank, label=label)
        self._draw_image(
            heatmap=self._merge_layer_smaps(
                self._create_layer_smaps(
                    self._create_channel_smaps(
                        self._create_weights(
                            n_samples=n_samples,
                            sigma=sigma,
                            random_state=random_state,
                        )
                    ),
                    channel_weight=channel_weight,
                    n_channels=n_channels,
                )
            ),
            draw_negative=draw_negative,
            fig=fig,
            ax=ax,
        )
        self._clear_cache()
        return
