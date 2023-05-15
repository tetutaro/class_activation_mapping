#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Optional, Any
import os

import numpy as np
import torch
from torch import Tensor
from PIL import Image, ImageFilter
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from cam.backbones.backbone import Backbone
from cam.base import (
    DEBUG,
    target_names,
    smooth_types,
    activation_weights,
    channel_weights,
    group_types,
    TargetLayer,
    batch_shape,
)
from cam.base.acquired import Acquired
from cam.base.context import Context
from cam.base.containers import SaliencyMaps
from cam.base.network_weight import NetworkWeight
from cam.base.activation_weight import ActivationWeight
from cam.base.channel_weight import ChannelWeight
from cam.base.layer_weight import LayerWeight
from cam.utils.display import draw_image_heatmap


class BaseCAM(NetworkWeight, ActivationWeight, ChannelWeight, LayerWeight):
    """The base class for all CAM models.

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
        backbone (Backbone): resource of the CNN model.
        batch_size (int): max number of images in a batch.
        n_divides (int): number of divides. (use it in IntegratedGrads)
        n_samples (int): number of samplings. (use it in SmoothGrad)
        sigma (float): sdev of Normal Dist. (use it in SmoothGrad)
        activation_weight (str): the type of weight for each activations.
        gradient_smooth (str): the method of smoothing gradient.
        gradient_no_gap (bool): if True, use gradient as is.
        channel_weight (str): the method of weighting for each channels.
        channel_group (str): the method of creating groups.
        channel_cosine (bool): if True, use cosine distance at clustering.
        channel_minmax (bool): if True, adopt the best&worst channel only.
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
        # settings for LayerWeight
        high_resolution: bool = False,
        # settings for CommonWeight
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        # initialize flags
        self.target_last_layer_: bool = False
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
            high_resolution=high_resolution,
            random_state=random_state,
            **backbone,
        )
        # check arguments
        self._check_arguments()
        # set flags
        if activation_weight == "class":
            self.target_last_layer_ = True
            self.set_class_weights(class_weights=self.get_class_weights())
        if channel_weight == "ablation":
            self.target_last_layer_ = True
        # set random seeds
        if self.random_state is not None:
            self._set_random_seeds()
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
        return target_layers

    # ## main function

    def draw(
        self: BaseCAM,
        path: str,
        target: TargetLayer,
        fig: Figure,
        ax: Axes,
        rank: Optional[int] = None,
        label: Optional[int] = None,
        draw_negative: bool = False,
        draw_colorbar: bool = False,
        title: Optional[str] = None,
        title_model: bool = False,
        title_label: bool = False,
        title_score: bool = False,
        **kwargs: Any,
    ) -> None:
        """the main function.

        Args:
            path (str): the pathname of the original image.
            target (TargetLayer): target Conv. Layers to retrieve activations.
            fig (Figure): the Figure instance.
            ax (Axes): the Axes instance.
            rank (Optional[int]): the rank of the target class.
            label (Optional[int]): the label of the target class.
            draw_negative (bool): draw negative regions or not.
            draw_colorbar (bool): draw colorbar.
            title (Optional[str]): title of heatmap.
            title_model (bool): show model name in title.
            title_label (bool): show label name in title.
            title_score (bool): show score in title.

        Raises:
            ValueError: parameters are invalid.
            IndexError: target is out of index.
            FileNotFoundError: path is not exists.
        """
        # convert target to List[str]
        target_layers: List[str] = self._convert_target(target=target)
        # load the original image
        if not os.path.exists(path=path):
            raise FileNotFoundError(f"{path} is not exists")
        raw_image: Image = Image.open(path)
        image: Tensor = self.transforms(raw_image).unsqueeze(0).to(self.device)
        if label is None:
            # decide the target label
            if rank is None:
                rank = 1
            if rank >= self.n_labels:
                raise ValueError(
                    f"rank ({rank}) is too large (< {self.n_labels})"
                )
            scores: Tensor = self.forward(image=image)
            if DEBUG:
                assert batch_shape(scores) == 1
            label = (
                scores.detach()
                .argsort(dim=1, descending=True)
                .squeeze()
                .cpu()
                .numpy()[rank]
            )
            del scores
        elif label >= self.n_labels:
            raise ValueError(
                f"label ({label}) is too large (< {self.n_labels})"
            )
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
        # normalize heatmap
        heatmap = heatmap / heatmap.max()
        if draw_negative:
            heatmap = heatmap.clip(min=-1.0, max=1.0)
        else:
            heatmap = heatmap.clip(min=0.0, max=1.0)
        # title
        if title is None:
            label_name: str = self.labels[ctx.label]
            if title_score:
                label_name += f" ({ctx.score:.4f})"
            if title_model:
                title = self.cam_name
                if title_label:
                    title += f" ({label_name})"
            elif title_label:
                title = label_name
        # draw the heatmap
        fig = draw_image_heatmap(
            image=ctx.raw_image,
            heatmap=heatmap,
            title=title,
            fig=fig,
            ax=ax,
            draw_negative=draw_negative,
            draw_colorbar=draw_colorbar,
        )
        # clear cache
        ctx.clear()
        return
