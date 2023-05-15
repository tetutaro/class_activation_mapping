#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""wrapper class for all CNN models (backbone) and all CAM models.
"""
from __future__ import annotations
from typing import Tuple, Dict, Optional, Any
import os
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from PIL import Image

from cam.backbones.backbone import Backbone
from cam.backbones import __all__ as all_backbones
from cam.base import (
    smooth_types,
    activation_weights,
    channel_weights,
    group_types,
    TargetLayer,
)
from cam.base.base_cam import BaseCAM
from cam.models import __all__ as all_cam_models
from cam.utils.display import show_text

# correct all backbone CNN models
backbones: Dict[str, Backbone] = dict()
for backbone in all_backbones:
    ldic: Dict[str, Backbone] = dict()
    source: str = f"""from cam.backbones import {backbone}
backbones[{backbone}["cnn_name"]] = {backbone}"""
    exec(source, globals(), ldic)
# correct all CAM models
cam_models: Dict[str, BaseCAM] = dict()
for cam_model in all_cam_models:
    ldic: Dict[str, BaseCAM] = dict()
    source: str = f"""from cam.models import {cam_model}
cam_models[{cam_model}.cam_name] = {cam_model}"""
    exec(source, globals(), ldic)
cam_models["Base-CAM"] = BaseCAM


class CAM:
    """wapper class for all CAM model class

    Args:
        cnn_model (str): the name of backbone CNN model.
        cam_model (str): the name of CAM model.
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
        backbones (Dict[str, Backbone]): all backbone resources.
        cam_models (Dict[str, BaseCAM]): all CAM model classes.
        cnn_model (str): the name of backbone CNN.
        cam_model (str): the name of CAM model.
        backbone (Backbone): the backbone resource.
        cam (BaseCAM): the CAM model.
    """

    backbones: Dict[str, Backbone] = backbones
    cam_models: Dict[str, BaseCAM] = cam_models
    cnn_model: str
    cam_model: str
    backbone: Backbone
    cam: BaseCAM

    def __init__(
        self: CAM,
        # name of backbone and CAM
        cnn_model: str,
        cam_model: str,
        # settings for NetworkWeight
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
        if cnn_model not in list(self.backbones.keys()):
            raise ValueError(f"{cnn_model} not exists")
        if cam_model not in list(self.cam_models.keys()):
            raise ValueError(f"{cam_model} not exists")
        self.cnn_model = cnn_model
        self.cam_model = cam_model
        self.backbone = self.backbones[cnn_model]
        self.cam = self.cam_models[cam_model](
            backbone=self.backbone,
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
        )
        return

    @classmethod
    def show_cnn_models(self: CAM) -> None:
        """list names of backbone.

        Returns:
            List[str]: names of backbone.
        """
        show_text("\n".join(["* " + x for x in self.backbones.keys()]))
        return

    @classmethod
    def show_cam_models(self: CAM) -> None:
        """list names of CAM models.

        Returns:
            List[str]: names of CAM models.
        """
        show_text("\n".join(["* " + x for x in self.cam_models.keys()]))
        return

    def set_cam_name(self: CAM, cam_name: str) -> None:
        """set the name of this CAM model as you like.

        Args:
            cam_name (str): the name of CAM model.
        """
        self.cam.set_cam_name(cam_name=cam_name)
        return

    def show_conv_layers(self: CAM) -> None:
        """show information of Conv. Layers"""
        self.cam.show_conv_layers()
        return

    def draw(
        self: CAM,
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
    ) -> None:
        """draw overlayed heatmap on the original image.

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
        """
        self.cam.draw(
            path=path,
            target=target,
            fig=fig,
            ax=ax,
            rank=rank,
            label=label,
            draw_negative=draw_negative,
            draw_colorbar=draw_colorbar,
            title=title,
            title_model=title_model,
            title_label=title_label,
            title_score=title_score,
        )
        return


def main() -> None:
    """ """
    # parse arguments
    parser: ArgumentParser = ArgumentParser(
        description="draw heatmap of attention of image"
    )
    # MODEL
    parser_model: ArgumentParser = parser.add_subparsers(
        title="MODEL",
        help="name of CNN and CAM",
    )
    parser_model.add_argument(
        "--cnn-model",
        type=str,
        required=True,
        choices=list(backbones.keys()),
        help="the name of backbone CNN model",
    )
    parser_model.add_argument(
        "--cam-model",
        type=str,
        required=True,
        choices=list(cam_models.keys()),
        help="the name of CAM model",
    )
    # CAM
    parser_cam: ArgumentParser = parser.add_subparsers(
        title="CAM",
        help="settings for CAM",
    )
    parser_cam.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="max number of images in a batch",
    )
    parser_cam.add_argument(
        "--n-divides",
        type=int,
        default=8,
        help="number of divides (use it in IntegratedGrads)",
    )
    parser_cam.add_argument(
        "--n-samples",
        type=int,
        default=8,
        help="number of samplings (use it in SmoothGrad)",
    )
    parser_cam.add_argument(
        "--sigma",
        type=float,
        default=0.3,
        help="sdev of Normal Dist (use it in SmoothGrad)",
    )
    parser_cam.add_argument(
        "--activation-weight",
        type=str,
        choices=activation_weights,
        default="none",
        help="the type of weight for each activations",
    )
    parser_cam.add_argument(
        "--gradient-smooth",
        type=str,
        choices=smooth_types,
        default="none",
        help="the method of smoothing gradient",
    )
    parser_cam.add_argument(
        "--gradient-no-gap", action="store_true", help="use gradient as is"
    )
    parser_cam.add_argument(
        "--channel-weight",
        type=str,
        choices=channel_weights,
        default="none",
        help="the method of weighting for each channels",
    )
    parser_cam.add_argument(
        "--channel-group",
        type=str,
        choices=group_types,
        help="the method of creating groups",
    )
    parser_cam.add_argument(
        "--channel-cosine",
        action="store_true",
        help="use cosine distance at clustering",
    )
    parser_cam.add_argument(
        "--channel-minmax",
        action="store_true",
        help="adopt the best & worst channel group only",
    )
    parser_cam.add_argument(
        "--n-channels",
        type=int,
        default=-1,
        help=(
            "the number of abscission channel groups to calculate"
            "(deafult: -1 (= all channel groups))"
        ),
    )
    parser_cam.add_argument(
        "--n-groups",
        type=int,
        default=None,
        help=(
            "number of channel groups"
            "(default: estimate automatically inside the CAM model)"
        ),
    )
    parser_cam.add_argument(
        "--high-resolution", action="store_true", help="high-reso heatmap"
    )
    parser_cam.add_argument(
        "--random-state", type=int, default=None, help="the random seed"
    )
    # INPUT
    parser_input: ArgumentParser = parser.add_subparsers(
        title="INPUT",
        help="settings for the input of CAM",
    )
    parser_input.add_argument(
        "--path", type=str, required=True, help="the pathname of the image"
    )
    parser_input.add_argument(
        "--target", type=TargetLayer, required=True, help="target Conv. Layer"
    )
    parser_input.add_argument(
        "--rank", type=int, default=None, help="the rank of the target class"
    )
    parser_input.add_argument(
        "--label", type=int, default=None, help="the label of the target class"
    )
    # OUTPUT
    parser_output: ArgumentParser = parser.add_subparsers(
        title="OUTPUT",
        help="settings for the output image",
    )
    parser_output.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "the pathname of heatmap overlayed image."
            "(default: heatmap_[PATH])"
        ),
    )
    parser_output.add_argument(
        "--draw-negative", action="store_true", help="draw negative regions"
    )
    parser_output.add_argument(
        "--draw-colorbar", action="store_true", help="draw colorbar"
    )
    parser_output.add_argument(
        "--figsize",
        type=Tuple[float, float],
        default=None,
        help=(
            "size of figure (width[100 dpi], height[100 dpi])"
            "(default: the same size of the original image)"
        ),
    )
    parser_output.add_argument(
        "--title",
        type=str,
        default=None,
        help=(
            "title of ouput image."
            "(if don't set, create automatically accordings to below flags)"
        ),
    )
    parser_output.add_argument(
        "--title-model", action="store_true", help="show model name in title"
    )
    parser_output.add_argument(
        "--title-label", action="store_true", help="show label name in title"
    )
    parser_output.add_argument(
        "--title-score", action="store_true", help="show score in title"
    )
    parser_output.add_argument(
        "--font-family", type=str, default="sans-serif", help="font family"
    )
    parser_output.add_argument(
        "--font-size", type=float, default=10.0, help="font size"
    )
    args: Namespace = parser.parse_args()
    # check arguments
    if not os.path.exists(path=args.path):
        raise FileNotFoundError(f"{args.path} is not exists")
    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(args.path),
            "heatmap_" + os.path.basename(args.path),
        )
    if args.figsize is None:
        image: Image = Image.open(args.path)
        args.figsize = (image.size[0] / 100, image.size[1] / 100)
    # create CAM
    cam: CAM = CAM(**vars(args))
    # customize font and axes
    # default settings
    # https://matplotlib.org/stable/tutorials/introductory/customizing.html#the-default-matplotlibrc-file
    rc: Dict[str, Any] = dict()
    rc["font.family"] = args.font_family
    rc["font.size"] = args.font_size
    rc["xtick.labelsize"] = args.font_size * 0.6
    rc["ytick.labelsize"] = args.font_size * 0.6
    rc["axes.linewidth"] = 0.2
    rc["xtick.major.size"] = 2.0
    rc["xtick.major.width"] = 0.2
    rc["ytick.major.size"] = 2.0
    rc["ytick.major.width"] = 0.2
    # draw heatmap
    with plt.rc_context(rc=rc):
        fig, ax = plt.subplots(figsize=args.figsize)
        cam.draw(fig=fig, ax=ax, **vars(args))
        fig.tight_layout()
        plt.savefig(args.output)
        plt.clf()
        plt.close()
    return


if __name__ == "__main__":
    main()
