#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""utility functions"""
from typing import Callable, Optional

import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from IPython import get_ipython
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from skimage.segmentation import mark_boundaries


sigmoid: Callable[[float], float] = lambda x: 1.0 / (1.0 + np.exp(-x))


def is_env_notebook() -> bool:
    """detect environment.

    Returns:
        bool: environment is JupyterNotebook.
    """
    try:
        env_name = get_ipython().__class__.__name__
        if env_name == "ZMQInteractiveShell":
            return True
    except Exception:
        pass
    return False


def show_text(text: str) -> None:
    """print the text to the notebook or the console.

    Args:
        text (str): strings to print.
    """
    if is_env_notebook():
        display(Markdown(text.strip()))
    else:
        print(text.strip())
    return


def show_table(df: pd.DataFrame, **kwargs) -> None:
    """print the table to the notebook or the console.

    Args:
        df (pd.DataFrame): table to print.
    """
    if is_env_notebook():
        display(Markdown(df.to_markdown(**kwargs)))
    else:
        print(df.to_markdown(tablefmt="simple", **kwargs))
    return


def draw_image_boundary(
    image: np.ndarray,
    boundary: np.ndarray,
    title: str,
    ax: Axes,
) -> None:
    """draw the original image and its boundaries of segments.

    Args:
        image (np.ndarray): the original image.
        boundary (np.ndarray): boundaries.
        titie (str): the title of the image.
        fig (Figure): the Figure instance.
        ax (Axes): the Axes instance.
    """
    ax.imshow(mark_boundaries(image, boundary))
    ax.set_title(title)
    ax.set_axis_off()
    return


def draw_image_heatmap(
    image: Image,
    heatmap: np.ndarray,
    title: Optional[str],
    fig: Figure,
    ax: Axes,
    draw_negative: bool = False,
    draw_colorbar: bool = False,
) -> None:
    """draw the original image and the heatmap over the original image.

    Args:
        image (np.ndarray): the original image.
        heatmap (np.ndarray): the heatmap.
        titie (Optional[str]): the title of the image.
        fig (Figure): the Figure instance.
        ax (Axes): the Axes instance.
        draw_negative (bool): draw negative regions.
        draw_colorbar (bool): draw colorbar.
    """
    # check values of heatmap
    assert heatmap.max() <= 1.0
    if not draw_negative:
        assert heatmap.min() >= 0.0
    else:
        assert heatmap.min() >= -1.0
    # draw original image
    ax.imshow(image)
    # draw overlay (heatmap)
    mappable: AxesImage
    if draw_negative:
        mappable = ax.imshow(
            heatmap, cmap="RdBu_r", vmin=-1.0, vmax=1.0, alpha=0.5
        )
    else:
        mappable = ax.imshow(
            heatmap, cmap="jet", vmin=0.0, vmax=1.0, alpha=0.5
        )
    if title is not None:
        ax.set_title(title)
    ax.set_axis_off()
    if draw_colorbar:
        fig.colorbar(mappable, ax=ax, shrink=0.8)
    return
