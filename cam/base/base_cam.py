#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor
from PIL import Image, ImageFilter
import matplotlib as mpl

from cam.backbones.backbone import Backbone
from cam.base import (
    DEBUG,
    target_names,
    smooth_types,
    activation_weights,
    position_weights,
    channel_weights,
    group_types,
    merge_layers,
    TargetLayer,
    batch_shape,
)
from cam.base.acquired import Acquired
from cam.base.context import Context
from cam.base.containers import SaliencyMaps
from cam.base.network import Network
from cam.base.raw_smaps import RawSMAPS
from cam.base.position_smaps import PositionSMAPS
from cam.base.channel_smaps import ChannelSMAPS
from cam.base.final_smap import FinalSMAP
from cam.utils.display import draw_image_heatmap


class BaseCAM(Network, RawSMAPS, PositionSMAPS, ChannelSMAPS, FinalSMAP):
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
        name (str): the name of this CAM model.
        backbone (Backbone): resource of the CNN model.
        activation_weight (str): the type of weights for each activations.
        gradient_gap (bool): average weights over channels or not.
        gradient_smooth (str): the method of smoothing gradients.
        position_weight (str): the method of weighting for each positions.
        position_group (str): the method of creating groups.
        channel_weight (str): the method of weighting for each channels.
        channel_group (bool): the method of creating groups.
        merge_layer (str):
            the method to merge channel saliency maps over layers.
        batch_size (int): max number of images to forward images at once.
        n_divides (int): number of divides. (use it in IntegratedGrads)
        n_samples (int): number of samplings. (use it in SmoothGrad)
        n_positions (int): the max number of culculate abscissions.
        n_position_groups (Optional[int]): the number of positional clusters.
        n_channels (int): the max number of culculate abscissions.
        n_channel_groups (Optional[int]): the number of channel clusters.
        sigma (float): sdev of Normal Dist. (use it in SmoothGrad)
        random_state (Optional[int]): the random seed.
    """

    def __init__(
        self: BaseCAM,
        name: str,
        # settings of CNN
        backbone: Backbone,
        # settings for raw saliency maps
        activation_weight: str = "none",
        gradient_smooth: str = "none",
        gradient_gap: bool = True,
        # settings for position saliency maps
        position_weight: str = "none",
        position_group: str = "none",
        # settings for channel saliency maps
        channel_weight: str = "none",
        channel_group: str = "none",
        # settings for (final) saliency map
        merge_layer: str = "none",
        # parameters
        batch_size: int = 8,
        n_divides: int = 8,
        n_samples: int = 8,
        n_positions: int = -1,
        n_position_groups: Optional[int] = None,
        n_channels: int = -1,
        n_channel_groups: Optional[int] = None,
        sigma: float = 0.3,
        random_state: Optional[int] = None,
    ) -> None:
        # initialize flags
        self.target_last_layer_: bool = False
        # initialize parents
        super().__init__(
            activation_weight=activation_weight,
            gradient_gap=gradient_gap,
            gradient_smooth=gradient_smooth,
            position_weight=position_weight,
            position_group=position_group,
            channel_weight=channel_weight,
            channel_group=channel_group,
            merge_layer=merge_layer,
            batch_size=batch_size,
            n_divides=n_divides,
            n_samples=n_samples,
            n_positions=n_positions,
            n_position_groups=n_position_groups,
            n_channels=n_channels,
            n_channel_groups=n_channel_groups,
            sigma=sigma,
            **backbone,
        )
        # set flags
        if activation_weight == "class":
            self.target_last_layer_ = True
            self.set_class_weights(class_weights=self.get_class_weights())
        if (position_weight == "ablation") or (channel_weight == "ablation"):
            self.target_last_layer_ = True
        # the name of CAM model
        self.name_ = name
        # set random seeds
        if random_state is not None:
            self._set_random_seeds(random_state=random_state)
        return

    def _set_random_seeds(self: BaseCAM, random_state: int) -> None:
        """set random seeds

        Args:
            random_state (int): the seed of random
        """
        # random.seed(random_state)
        np.random.seed(seed=random_state)
        torch.manual_seed(seed=random_state)
        torch.cuda.manual_seed(seed=random_state)
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
        if self.position_weight not in position_weights:
            _raise_error(name="position_weight")
        if self.position_group not in group_types:
            _raise_error(name="position_group")
        if self.channel_weight not in channel_weights:
            _raise_error(name="channel_weight")
        if self.channel_group not in group_types:
            _raise_error(name="channel_group")
        if self.merge_layer not in merge_layers:
            _raise_error(name="merge_layer")
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
        rank: Optional[int] = None,
        label: Optional[int] = None,
        draw_negative: bool = False,
        fig: Optional[mpl.figure.Figure] = None,
        ax: Optional[mpl.axes.Axes] = None,
    ) -> None:
        """the main function.

        Args:
            path (str): the pathname of the original image.
            target (TargetLayer): target Conv. Layers to retrieve activations.
            rank (Optional[int]): the rank of the target class.
            label (Optional[int]): the label of the target class.
            draw_negative (bool): draw negative regions or not.
            fig (Optional[mpl.figure.Figure]):
                the Figure instance that the output image is drawn.
            ax (Optinonal[mpl.axies.Axes):
                the Axes instance that the output image is drawn.
        """
        # convert target to List[str]
        target_layers: List[str] = self._convert_target(target=target)
        # load the original image
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
        # -> raw saliency maps
        raw_smaps: SaliencyMaps = self.create_raw_smaps(ctx=ctx)
        # weight position-wise weights to raw salincy maps
        # -> position saliency maps
        position_smaps: SaliencyMaps = self.create_position_smaps(
            raw_smaps=raw_smaps, ctx=ctx
        )
        # weight channel-wise weights and merge over channels
        # -> channel saliency maps
        channel_smaps: SaliencyMaps = self.create_channel_smaps(
            position_smaps=position_smaps, ctx=ctx
        )
        # merge channel saliency maps over layers
        # -> final saliency map
        final_smap: Tensor = self.create_final_smap(
            channel_smaps=channel_smaps, ctx=ctx
        )
        # convert final saliency map (Tensor) to heatmap (numpy.ndarray)
        heatmap: np.ndarray = final_smap.detach().cpu().numpy()
        if DEBUG:
            assert heatmap.shape == (ctx.height, ctx.width)
        del final_smap
        # normalize heatmap
        heatmap = heatmap / heatmap.max()
        if draw_negative:
            heatmap = heatmap.clip(min=-1.0, max=1.0)
        else:
            heatmap = heatmap.clip(min=0.0, max=1.0)
        # title
        title: str = f"{self.name_}"
        if fig is None or ax is None:
            title += f" ({self.labels[ctx.label]})"
        # draw the heatmap
        draw_image_heatmap(
            image=ctx.raw_image,
            heatmap=heatmap,
            title=title,
            draw_negative=draw_negative,
            fig=fig,
            ax=ax,
        )
        # clear cache
        ctx.clear()
        return
