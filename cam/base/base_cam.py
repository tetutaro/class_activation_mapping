#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Optional, Any
import os

import numpy as np
import torch
from torch import Tensor
from PIL import Image, ImageFilter
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

from cam.backbones.backbone import Backbone
from cam.base import (
    DEBUG,
    target_names,
    smooth_types,
    activation_weights,
    channel_weights,
    group_types,
    TargetLayer,
)
from cam.base.acquired import Acquired
from cam.base.context import Context
from cam.base.containers import SaliencyMaps
from cam.base.network_weight import NetworkWeight
from cam.base.activation_weight import ActivationWeight
from cam.base.channel_weight import ChannelWeight
from cam.base.layer_weight import LayerWeight
from cam.utils.display import (
    sigmoid,
    draw_image_heatmap,
    draw_histogram,
    draw_inertias,
)


class BaseCAM(NetworkWeight, ActivationWeight, ChannelWeight, LayerWeight):
    """The base class for all CAM models.

    The aim of CAM models is to create a "Saliency Map" that represents
    the discriminative regions of the original image
    (the region that the CNN model pays attention (or doesn't pay attention)
    to determine the indicated label).

    To create the saliency map,
    CAM models retrieve "Activation"
    (output of Conv. Layer when forward the original image to the CNN),
    weight them with some weight,
    and merge them over channels and layers.
    (Sometimes "Activation" is named as "Feature Map".)

    One of the strong candidates for the weight of activations is
    "Gradient" (output of Conv. Layer when backword the score to the CNN).

    There are following 4 processes to create the saliency map.

    #. forward the image to CNN and get activations, scores and gradients.
        * -> cam.base.network_weight.NetworkWeight.acquires_gred()
    #. create weights for activations from Gradients (or class weight)
        * -> cam.base.activation_weight.ActivationWeight.weight_activation()
    #. group (cluster) channels and weight for each channel (group)
        * -> cam.base.channel_weight.ChannelWeight.weight_channel()
    #. merge layers
        * -> cam.base.layer_weight.LayerWeight.weight_layer()

    Args:
        backbone (Backbone): resource of the CNN model.
        batch_size (int): max number of images in a batch.
        n_divides (int): number of divides. (use it in IntegratedGrads)
        n_samples (int): number of samplings. (use it in SmoothGrad)
        sigma (float): sdev of Normal Dist. (use it in SmoothGrad)
        activation_weight (str): the type of weight for each activation.
        gradient_smooth (str): the method of smoothing gradient.
        gradient_no_gap (bool): if True, use gradient as is.
        channel_weight (str): the method of weighting for each channel.
        channel_group (str): the method of creating groups.
        channel_cosine (bool): if True, use cosine distance at clustering.
        channel_minmax (bool): if True, adopt the best&worst channel only.
        normalize_softmax (bool): normalize abscission score using softmax.
        n_channels (int): the number of abscission channel groups to calc.
        n_groups (Optional[int]): the number of channel groups.
        high_resolution (bool): if True, produce high resolution heatmap.
        random_state (Optional[int]): the random seed.

    Attributes:
        cam_name (str): the name of this CAM model.
    """

    cam_name: str = "Base-CAM"

    def __init__(
        self: BaseCAM,
        # settings for NetworkWeight
        backbone: Backbone,
        batch_size: int = 8,
        n_divides: int = 8,
        n_samples: int = 8,
        sigma: float = 0.3,
        # settings for ActivationWeight
        activation_weight: str = "none",
        gradient_smooth: str = "none",
        gradient_no_gap: bool = False,
        # settings for ChannelWeight
        channel_weight: str = "none",
        channel_group: str = "none",
        n_channels: int = -1,
        n_groups: Optional[int] = None,
        channel_cosine: bool = False,
        channel_minmax: bool = False,
        normalize_softmax: bool = False,
        # settings for LayerWeight
        high_resolution: bool = False,
        # settings for CommonWeight
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        # the target layer have to be the last Conv. Layer
        self.target_last_layer_: bool = False
        # the number of target layer have to be 1
        self.target_one_layer_: bool = False
        # initialize parents
        super().__init__(
            batch_size=batch_size,
            n_divides=n_divides,
            n_samples=n_samples,
            sigma=sigma,
            activation_weight=activation_weight,
            gradient_smooth=gradient_smooth,
            gradient_no_gap=gradient_no_gap,
            channel_weight=channel_weight,
            channel_group=channel_group,
            n_channels=n_channels,
            n_groups=n_groups,
            channel_cosine=channel_cosine,
            channel_minmax=channel_minmax,
            normalize_softmax=normalize_softmax,
            high_resolution=high_resolution,
            random_state=random_state,
            **backbone,
        )
        # check arguments
        self._check_arguments()
        # set flags
        if activation_weight == "class":
            self.target_last_layer_ = True
            self.set_class_weight(class_weight=self.get_class_weight())
        if channel_weight == "ablation":
            self.target_last_layer_ = True
        if channel_group != "none":
            self.target_one_layer_ = True
        # set random seeds
        if self.random_state is not None:
            self._set_random_seeds()
        # hidden flag
        self.draw_distribution_: bool = False
        self.draw_inertias_: bool = False
        return

    def set_cam_name(self: BaseCAM, cam_name: str) -> None:
        """set the name of this CAM model as you like.

        Args:
            cam_name (str): the name of CAM model.
        """
        self.cam_name = cam_name
        return

    def _set_random_seeds(self: BaseCAM) -> None:
        """set random seeds

        Args:
            random_state (int): the seed of random
        """
        # random.seed(self.random_state)
        np.random.seed(seed=self.random_state)
        torch.manual_seed(seed=self.random_state)
        torch.cuda.manual_seed(seed=self.random_state)
        # torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms = True
        return

    # ## check functions

    def _check_arguments(self: BaseCAM) -> None:
        """check arguments.

        Raises:
            ValueError: arguments in invalid.
        """

        def _raise_error(name: str) -> None:
            value: str = getattr(self, name)
            raise ValueError(f"invalid {name}: {value}")

        if self.activation_weight not in activation_weights:
            _raise_error(name="avtivation_weight")
        if self.gradient_smooth not in smooth_types:
            _raise_error(name="gradient_smooth")
        if self.channel_weight not in channel_weights:
            _raise_error(name="channel_weight")
        if self.channel_group not in group_types:
            _raise_error(name="channel_group")
        return

    def _assert_rank_label(
        self: BaseCAM,
        rank: Optional[int],
        label: Optional[int],
    ) -> Optional[int]:
        """assert label and rank.

        Args:
            label (Optional[int]): label.
            rank (Optional[int]): rank.

        Returns:
            Optional[int]: rank.

        Raises:
            ValueError: too large rank/label.
        """
        if label is None:
            if rank is None:
                rank = 1
            if rank >= self.n_labels:
                raise ValueError(
                    f"rank ({rank}) is too large (< {self.n_labels})"
                )
        elif label >= self.n_labels:
            raise ValueError(
                f"label ({label}) is too large (< {self.n_labels})"
            )
        return rank

    def _convert_target(self: BaseCAM, target: TargetLayer) -> None:
        """check target and convert target from TargetLayer to List[str]

        Args:
            target (TargetLayer): the target layer(s).

        Returns:
            List[str]: converted target.

        Raises:
            ValueError: target is invalid.
            IndexError: index is out of range.
        """
        target_layers: List[str]
        # check target value
        if isinstance(target, str):
            if target not in target_names:
                raise ValueError(f"invalid target: {target}")
            if target == "last":
                target_layers = [self.conv_layers[-1]]
            else:  # target == "all"
                target_layers = self.conv_layers[:]
        elif isinstance(target, int):
            name: str = self.conv_layers[target]
            target_layers = [name]
        elif isinstance(target, list):
            n_conv_layers: int = len(self.conv_layers)
            target_vals: List[int] = list()
            for t in target:
                if not isinstance(t, int):
                    raise ValueError("invalid type target: {t}")
                _ = self.conv_layers[t]
                if t >= 0:
                    target_vals.append(t)
                else:
                    target_vals.append(n_conv_layers + t)
            target_vals = np.unique(np.array(target_vals)).tolist()
            target_layers = [self.conv_layers[x] for x in target_vals]
        else:
            raise ValueError("invalid type target: {target}")
        # check target is the last layer
        if self.target_last_layer_:
            if len(target_layers) > 1 or (
                target_layers[0] != self.conv_layers[-1]
            ):
                raise ValueError('target should be "last"')
        # check number of target is 1
        if self.target_one_layer_:
            if len(target_layers) > 1:
                raise ValueError("the number of target should be 1")
        return target_layers

    def _set_title(
        self: BaseCAM,
        title: Optional[str],
        title_model: bool,
        title_label: bool,
        title_score: bool,
        ctx: Context,
    ) -> Optional[str]:
        """set title.

        Args:
            title (Optional[str]): title of heatmap.
            title_model (bool): show model name in title.
            title_label (bool): show label name in title.
            title_score (bool): show score in title.
            ctx (Context): context of this process.

        Returns:
            Optional[str]: title.
        """
        if title is not None:
            return title
        label_name: str = self.labels[ctx.label]
        if len(label_name) > 12:
            label_names: List[str]
            if " " in label_name:
                label_names = label_name.split()
            else:
                label_names = label_name.split("-")
            label_name = " ".join(
                [x[0].upper() + "." for x in label_names[:-1]]
            )
            label_name += " " + label_names[-1]
        if title_score:
            label_name += f" ({sigmoid(ctx.score):.4f})"
        if title_model:
            title = self.cam_name
            if title_label:
                title += f" ({label_name})"
        elif title_label:
            title = label_name
        return title

    def _set_xlabel(self: BaseCAM) -> Optional[str]:
        """set xlabel

        Returns:
            Optional[str]: xlabel
        """
        xlabel: Optional[str] = None
        if self.n_groups_ is not None:
            xlabel = f"# of clusters = {self.n_groups_}"
        return xlabel

    # ## main function

    def draw(
        self: BaseCAM,
        path: str,
        target: TargetLayer,
        ax: Axes,
        rank: Optional[int] = None,
        label: Optional[int] = None,
        draw_negative: bool = False,
        title: Optional[str] = None,
        title_model: bool = False,
        title_label: bool = False,
        title_score: bool = False,
        **kwargs: Any,
    ) -> Optional[AxesImage]:
        """the main function.

        Args:
            path (str): the pathname of the original image.
            target (TargetLayer): target Conv. Layers to retrieve activations.
            ax (Axes): the Axes instance.
            rank (Optional[int]): the rank of the target class.
            label (Optional[int]): the label of the target class.
            draw_negative (bool): draw negative regions or not.
            title (Optional[str]): title of heatmap.
            title_model (bool): show model name in title.
            title_label (bool): show label name in title.
            title_score (bool): show score in title.

        Returns:
            Optional[AxesImage]: colorbar.

        Raises:
            ValueError: parameters are invalid.
            IndexError: target is out of index.
            FileNotFoundError: path is not exists.
        """
        # assert label and rank
        rank = self._assert_rank_label(rank=rank, label=label)
        # convert target to List[str]
        target_layers: List[str] = self._convert_target(target=target)
        # load the original image
        if not os.path.exists(path=path):
            raise FileNotFoundError(f"{path} is not exists")
        raw_image: Image = Image.open(path)
        image: Tensor = self.transforms(raw_image).unsqueeze(0).to(self.device)
        # decide the target label
        if label is None:
            scores: Tensor = self.forward(image=image)
            label = (
                scores.detach()
                .argsort(dim=1, descending=True)
                .squeeze()
                .cpu()
                .numpy()[rank]
            )
            del scores
        # forward network
        acquired: Acquired = self.acquires_grad(
            target_layers=target_layers,
            image=image,
            label=label,
            smooth=self.gradient_smooth,
        )
        # create context
        ctx: Context = Context(
            forward_fn=self.forward,
            classify_fn=self.classify,
            raw_image=raw_image,
            width=raw_image.size[0],
            height=raw_image.size[1],
            image=image,
            blurred_image=(
                self.transforms(
                    raw_image.filter(
                        filter=ImageFilter.GaussianBlur(radius=51)
                    )
                )
                .unsqueeze(0)
                .to(self.device)
            ),
            label=label,
            score=acquired.scores.detach().squeeze().cpu().numpy()[label],
            activations=acquired.activations.clone(),
            gradients=acquired.gradients.clone(),
        )
        acquired.clear()
        # weight the weights made from gradients (etc...) to activations
        smaps: SaliencyMaps = self.weight_activation(ctx=ctx)
        # weight channel-wise weights and merge over channels
        smaps = self.weight_channel(smaps=smaps, ctx=ctx)
        # merge over layers
        smap: Tensor = self.weight_layer(smaps=smaps, ctx=ctx)
        # convert final saliency map (Tensor) to heatmap (numpy.ndarray)
        heatmap: np.ndarray = smap.detach().cpu().numpy()
        if DEBUG:
            assert heatmap.shape == (ctx.height, ctx.width)
        del smap
        # title
        title = self._set_title(
            title=title,
            title_model=title_model,
            title_label=title_label,
            title_score=title_score,
            ctx=ctx,
        )
        # xlabel
        xlabel: Optional[str] = self._set_xlabel()
        # hidden output
        if self.draw_distribution_:
            draw_histogram(dist=heatmap, ax=ax, title=title)
            ctx.clear()
            return
        if self.draw_inertias_:
            draw_inertias(
                inertias=self.inertias_,
                n_clusters=self.n_groups_,
                ax=ax,
                title=title,
            )
            ctx.clear()
            return
        # normalize heatmap
        heatmap = heatmap / heatmap.max()
        if draw_negative:
            heatmap = heatmap.clip(min=-1.0, max=1.0)
        else:
            heatmap = heatmap.clip(min=0.0, max=1.0)
        # draw the heatmap
        colorbar: AxesImage = draw_image_heatmap(
            image=ctx.raw_image,
            heatmap=heatmap,
            ax=ax,
            title=title,
            xlabel=xlabel,
            draw_negative=draw_negative,
        )
        # clear cache
        ctx.clear()
        return colorbar
