#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""utility functions"""
from typing import Optional

import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from IPython import get_ipython
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries


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
    fig: Optional[mpl.figure.Figure] = None,
    ax: Optional[mpl.axes.Axes] = None,
) -> None:
    """draw the original image and its boundaries of segments.

    Args:
        image (np.ndarray): the original image.
        boundary (np.ndarray): boundaries.
        titie (str): the title of the image.
        fig (Optional[mpl.figure.Figure]):
            the Figure instance that the output image is drawn.
            if None, create it inside this function.
        ax (Optinonal[mpl.axies.Axes):
            the Axes instance that the output image is drawn.
            if None, create it inside this function.
    """
    show: bool = False
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        show = True
    ax.imshow(mark_boundaries(image, boundary))
    ax.set_title(title)
    ax.set_axis_off()
    if show:
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close()
    return


def draw_image_heatmap(
    image: Image,
    heatmap: np.ndarray,
    title: str,
    draw_negative: bool = False,
    fig: Optional[mpl.figure.Figure] = None,
    ax: Optional[mpl.axes.Axes] = None,
) -> None:
    """draw the original image and the heatmap over the original image.

    Args:
        image (np.ndarray): the original image.
        heatmap (np.ndarray): the heatmap.
        titie (str): the title of the image.
        draw_negative (bool): draw negative regions.
        fig (Optional[mpl.figure.Figure]):
            the Figure instance that the output image is drawn.
            if None, create it inside this function.
        ax (Optinonal[mpl.axies.Axes):
            the Axes instance that the output image is drawn.
            if None, create it inside this function.
    """
    # check values of heatmap
    assert heatmap.max() <= 1.0
    if not draw_negative:
        assert heatmap.min() >= 0.0
    else:
        assert heatmap.min() >= -1.0
    # create fig, ax if not exists
    show: bool = False
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        show = True
    # draw original image
    ax.imshow(image)
    # draw overlay (heatmap)
    mappable: mpl.image.AxesImage
    if draw_negative:
        mappable = ax.imshow(
            heatmap, cmap="RdBu_r", vmin=-1.0, vmax=1.0, alpha=0.5
        )
    else:
        mappable = ax.imshow(
            heatmap, cmap="jet", vmin=0.0, vmax=1.0, alpha=0.5
        )
    ax.set_title(title)
    ax.set_axis_off()
    if show:
        fig.colorbar(mappable, ax=ax)
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close()
    return
