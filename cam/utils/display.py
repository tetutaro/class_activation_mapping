#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""utility functions"""
from typing import List, Dict, Callable, Optional

import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from IPython import get_ipython
from PIL import Image
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from skimage.segmentation import mark_boundaries
import seaborn as sns


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
    ax: Axes,
    title: Optional[str],
    xlabel: Optional[str] = None,
    draw_negative: bool = False,
) -> AxesImage:
    """draw the original image and the heatmap over the original image.

    Args:
        image (np.ndarray): the original image.
        heatmap (np.ndarray): the heatmap.
        ax (Axes): the Axes instance.
        titie (Optional[str]): the title of the image.
        xlabel (Optional[str]): the label of x-axis of the image.
        draw_negative (bool): draw negative regions.
    """
    # check values of heatmap
    assert heatmap.max() <= 1.0
    if not draw_negative:
        assert heatmap.min() >= 0.0
    else:
        assert heatmap.min() >= -1.0
    # draw original image
    _ = ax.imshow(image)
    # draw overlay (heatmap)
    colorbar: AxesImage
    if draw_negative:
        colorbar = ax.imshow(
            heatmap, cmap="RdBu_r", vmin=-1.0, vmax=1.0, alpha=0.5
        )
    else:
        colorbar = ax.imshow(
            heatmap, cmap="jet", vmin=0.0, vmax=1.0, alpha=0.5
        )
    if title is not None:
        ax.set_title(title)
    if xlabel is None:
        ax.set_axis_off()
    else:
        ax.set_xlabel(xlabel=xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
    return colorbar


def draw_histogram(
    dist: np.ndarray,
    ax: Axes,
    title: Optional[str] = None,
) -> None:
    """draw elbow point of inertias

    Args:
        dist (np.ndarray): distribution.
        ax (Axes): the Axes instance.
        title (Optional[str]): the title.
    """
    sns.histplot(data=dist.reshape(-1), stat="probability", bins=100, ax=ax)
    if title is not None:
        ax.set_title(title)
    return


def draw_inertias(
    inertias: Dict[str, List[float]],
    n_clusters: int,
    ax: Axes,
    title: Optional[str] = None,
) -> None:
    """draw elbow point of inertias

    Args:
        inertias (Dict[str, List[float]]): inertias,
        n_clusters (int): elbow point.
        fig (Figure): the Figure instance.
        ax (Axes): the Axes instance.
        title (Optional[str]): the title.
    """
    assert "inertia" in list(inertias.keys())
    assert "n_clusters" in list(inertias.keys())
    pd.DataFrame(inertias).plot(
        kind="line", x="n_clusters", y="inertia", ax=ax
    )
    if n_clusters in inertias["n_clusters"]:
        off: int = inertias["n_clusters"].index(n_clusters)
        val: float = inertias["inertia"][off]
        ax.plot(
            [n_clusters],
            [val],
            marker="o",
            markersize=10,
            markerfacecolor="red",
        )
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(xlabel="# of clusters")
    ax.set_ylabel(ylabel="inertia")
    ax.legend().remove()
    return
