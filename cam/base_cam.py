#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Optional
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
        n_channels (Optional[int]): the number of channels calc score.
    """

    def __init__(
        self: BaseCAM,
        resource: ResourceCNN,
        target: TargetLayer,
        requires_grad: bool = False,
        blur_image: bool = False,
        channel_weight: Optional[str] = None,
        n_channels: Optional[int] = None,
    ) -> None:
        super().__init__(target=target, **resource)
        # store flags
        self.requires_grad: bool = requires_grad
        self.blur_image: bool = blur_image
        self.channel_weight: Optional[str] = channel_weight
        self.n_channels: Optional[int] = n_channels
        # caches created in self._forward()
        self.raw_image: Optional[Image] = None
        self.image: Optional[Tensor] = None
        self.blurred_image: Optional[Tensor] = None
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.target_class: Optional[int] = None
        self.score: Optional[float] = None
        # constants
        self.eps = 1e-6
        # the name of CAM model
        self.name: str
        self._set_name()
        return

    @abstractmethod
    def _set_name(self: BaseCAM) -> None:
        """set the name of CAM model for the title of the output image."""
        self.name = ""
        return

    def _assert_target_is_last(self: BaseCAM, target: TargetLayer) -> None:
        if isinstance(target, str):
            if target == "last":
                return
        elif isinstance(target, int):
            if target == -1:
                return
        elif isinstance(target, list):
            if len(target) == 1 and target[0] == -1:
                return
        raise ValueError(f'target must be "last": {target}')

    def _create_image(
        self: BaseCAM,
        path: str,
    ) -> None:
        """load image and create tensors.

        Args:
            path (str): the pathname of the original image.
        """
        # load the original image
        self.raw_image = Image.open(path)
        self.width, self.height = self.raw_image.size
        self.image = (
            self.transform(self.raw_image).unsqueeze(0).to(self.device)
        )
        # create the blurred image
        if self.blur_image:
            self.blurred_image: Tensor = (
                self.transform(
                    self.raw_image.filter(
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
            rank (Optional[int]): the rank of the target class.
            label (Optional[int]): the label of the target class.
        """
        # if image and requires_grad are not indicated, use default one
        if image is None:
            image = self.image
        if requires_grad is None:
            requires_grad = self.requires_grad
        # check indicated rank and label
        if self.target_class is None:
            if rank is None and label is None:
                rank = 0
            if (rank is not None and rank >= 1000) or (
                label is not None and label >= 1000
            ):
                raise ValueError("rank and label must be less than 1000")
        # clear activations and gradients
        self.activations.clear()
        self.gradients.clear()
        # forward the image to the network
        if requires_grad:
            scores: Tensor = self._forward_net(image=image)
        else:
            with torch.no_grad():
                scores: Tensor = self._forward_net(image=image)
        # decide the target class
        if self.target_class is None:
            if label is not None:
                self.target_class = label
            else:
                self.target_class = (
                    scores.clone()
                    .detach()
                    .argsort(dim=1, descending=True)
                    .squeeze()
                    .cpu()
                    .numpy()[rank]
                )
        # get the score of target class
        self.score = (
            scores.clone().detach().squeeze().cpu().numpy()[self.target_class]
        )
        # backward the score to the network
        if requires_grad:
            self.net.zero_grad()
            scores[:, [self.target_class]].backward(retain_graph=None)
        # finalize activations and gradients
        self.activations.finalize()
        self.gradients.finalize()
        return

    @abstractmethod
    def _create_weights(self: BaseCAM) -> Weights:
        """create the weight.

        Returns:
            Weights: the weight.
        """
        weights: Weights = Weights()
        weights.finalize()
        return weights

    def _create_dummy_weights(self: BaseCAM) -> Weights:
        """create the dummy weight (each value of weight = 1).

        Returns:
            Weights: the dummy weight.
        """
        weights: Weights = Weights()
        for _, (_, k, _, _) in self.activations:
            weights.append(torch.ones((1, k, 1, 1)).to(self.device))
        weights.finalize()
        return weights

    def _extract_class_weights(self: BaseCAM) -> Weights:
        """extract weights of the target class from self.class_weight.

        If this function is used, the target of Conv. Layer must be the "last".

        Returns:
            Weights: the weight of the target class.
        """
        weights: Weights = Weights()
        weights.append(
            self.class_weight[[self.target_class], :].view(
                1, self.class_weight.size()[1], 1, 1
            )
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
        assert len(weights) == len(self.activations)
        for (weight, _), (activation, _) in zip(weights, self.activations):
            channel_saliencies.append(
                F.interpolate(
                    weight * activation,
                    size=(self.height, self.width),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        channel_saliencies.finalize()
        return channel_saliencies

    def _calc_eigen_weight(
        self: BaseCAM,
        smap: Tensor,
    ) -> Tensor:
        """calc a weight for each channels using SVD
        (Singular Value Decomposition).

        Args:
            smap (Tensor): the saliency map to calc weight.

        Returns:
            Tensor: the weight of channels.
        """
        assert batch_size(smap) == 1
        k = channel_size(smap)
        if k == 1:
            return torch.tensor([1.0]).view(1, 1, 1, 1).to(self.device)
        # create channel (k) x position (u x v) matrix
        CP: Tensor = smap.view(k, -1)
        # standarize
        CP_std: Tensor = (CP - CP.mean()) / CP.std()
        # SVD (singular value decomposition)
        Ch, SS, _ = torch.linalg.svd(CP_std, full_matrices=False)
        # get the first eigenvector of the channel feature matrix and normalize
        weight: Tensor = F.normalize(Ch.real[:, SS.argmax()], dim=0).view(
            1, k, 1, 1
        )
        # the eigen-vector may have the opposite sign
        if (weight * F.relu(smap)).sum() < 0:
            weight = -weight
        return weight.view(1, k, 1, 1)

    def _calc_score_weight(
        self: BaseCAM,
        smap: Tensor,
        n_channels: Optional[int],
    ) -> Tensor:
        """calc a weight for each channels using SVD
        (Singular Value Decomposition).

        Args:
            smap (Tensor): the saliency map to calc weight.
            n_channels (Optional[int]): the number of channels calc score.

        Returns:
            Tensor: the weight of channels.
        """
        assert batch_size(smap) == 1
        k = channel_size(smap)
        if k == 1:
            return torch.tensor([1.0]).view(1, 1, 1, 1).to(self.device)
        if n_channels is None:
            n_channels = -1
        orig_score: float = self.score
        dict_smaps: List[Dict] = list()
        for i in range(k):
            # extract i-th channel from channel saliency map
            raw_smap: Tensor = smap.clone().detach()[:, [i], :, :]
            # normalize
            smax: float = raw_smap.max().cpu().numpy().ravel()[0]
            smin: float = raw_smap.min().cpu().numpy().ravel()[0]
            sdif: float = smax - smin
            if sdif < self.eps:
                continue
            normed_smap: Tensor = (raw_smap - smin) / sdif
            # stack maps
            dict_smaps.append(
                {
                    "channel": i,
                    "key": sdif,
                    "raw": raw_smap,
                    "norm": normed_smap,
                }
            )
        if n_channels > 0:
            dict_smaps = sorted(
                dict_smaps, key=lambda x: x["key"], reverse=True
            )[:n_channels]
        # calc score and weights
        weights: List[float] = list()
        for dict_smap in dict_smaps:
            # create feature emphasized image
            mapped_image: Tensor = dict_smap["norm"] * self.image
            if self.blur_image:
                mapped_image += (1.0 - dict_smap["norm"]) * self.blurred_image
            # forward network
            self._forward(
                image=self.image * dict_smap["norm"],
                requires_grad=False,
            )
            weights.append(self.score - orig_score)
        # normalize weight
        # The paper of Score-CAM use softmax for normalization.
        # But softmax turns negative values into positive values.
        # For that reason, if the target class is not the first rank,
        # the positive region of saliency map can't show the target.
        # So, I use normalization with L2-norm instead of softmax.
        normed_weight: Tensor = F.normalize(
            torch.tensor(weights).to(self.device), dim=0
        )
        # create final weights
        weight: Tensor
        if n_channels > 0:
            idx: Tensor = torch.tensor([x["channel"] for x in dict_smaps]).to(
                self.device
            )
            weight = torch.zeros(k).scatter(
                dim=0, index=idx, src=normed_weight
            )
        else:
            weight = normed_weight
        # restore the score
        self.score = orig_score
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
        if self.channel_weight is not None:
            channel_weight = self.channel_weight
        if self.n_channels is not None:
            n_channels = n_channels
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
                # calc weight for channels using SVD
                weight = self._calc_eigen_weight(
                    smap=channel_smap,
                )
            else:  # channel_weight == "score"
                # calc weight for channels by calc scores
                weight = self._calc_score_weight(
                    smap=channel_smap,
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
        title: str = f"{self.name}"
        if fig is None or ax is None:
            title += f" ({self.labels[self.target_class]})"
        # draw
        draw_image_heatmap(
            image=self.raw_image,
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
        self.raw_image = None
        self.image = None
        self.blurred_image = None
        self.width = None
        self.height = None
        self.target_class = None
        self.score = None
        return

    def draw(
        self: BaseCAM,
        path: str,
        rank: Optional[int] = None,
        label: Optional[int] = None,
        channel_weight: Optional[str] = None,
        n_channels: Optional[int] = None,
        draw_negative: bool = False,
        fig: Optional[mpl.figure.Figure] = None,
        ax: Optional[mpl.axes.Axes] = None,
    ) -> None:
        """the main function.

        Args:
            path (str): the pathname of the original image.
            rank (Optional[int]): the rank of the target class.
            label (Optional[int]): the label of the target class.
            eigen_weight (bool):
                if True, weighted activation(s) are further weighted
                by the first eigenvector of decomposed matrix by SVD.
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
                    self._create_channel_smaps(self._create_weights()),
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
