#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Optional, NamedTuple, Any
import os

import numpy as np
from PIL import Image
from lime.lime_image import LimeImageExplainer, ImageExplanation
import torch
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

from cam.base.network_weight import NetworkWeight
from cam.backbones.backbone import Backbone
from cam.utils.display import (
    sigmoid,
    show_text,
    draw_image_boundary,
    draw_image_heatmap,
)


class PredLabel(NamedTuple):
    label: int
    name: str
    rank: int
    score: float

    def __str__(self: PredLabel) -> str:
        return (
            f"* {self.rank + 1}: "
            f"{self.name} "
            f"(label={self.label}) "
            f"(score={self.score:.4f})"
        )


class LimeImage(NetworkWeight):
    """how to use the LimeImageExplainer.

    Args:
        backbone (Backbone): the backbone CNN model.
        path (str): the pathname of the original image.
        top_labels (int): number of top labels to predict.
        num_features (int): number of features.
        num_samples (int): number of samplings.
        batch_size (int): max number of images in a batch.
        n_divides (int): number of divides. (use it in IntegratedGrads)
        n_samples (int): number of samplings. (use it in SmoothGrad)
        sigma (float): sdev of Normal Dist. (use it in SmoothGrad)
        ramdom_state (int): the random seed.
    """

    def __init__(
        self: LimeImage,
        backbone: Backbone,
        path: str,
        top_labels: int = 5,
        num_features: int = 100000,
        num_samples: int = 1000,
        batch_size: int = 8,
        n_divides: int = 8,
        n_samples: int = 8,
        sigma: float = 0.3,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            n_divides=n_divides,
            n_samples=n_samples,
            sigma=sigma,
            random_state=random_state,
            **backbone,
        )
        if not os.path.exists(path=path):
            raise FileNotFoundError(f"{path} is not found")
        # load image
        images: np.ndarray = np.array(Image.open(path))[np.newaxis, :]
        # predict labels
        self.top_labels_: int = top_labels
        self._predict_labels(images=images)
        # create explanation
        explainer: LimeImageExplainer = LimeImageExplainer(
            random_state=random_state
        )
        self.explain_: ImageExplanation = explainer.explain_instance(
            image=images[0],
            classifier_fn=self._predict,
            labels=self.labels,
            top_labels=self.top_labels_,
            num_features=num_features,
            num_samples=num_samples,
            batch_size=batch_size,
            hide_color=0,
            random_seed=random_state,
        )
        self.n_groups_: Optional[int] = None  # for compatibility
        return

    def _predict(self: LimeImage, images: np.ndarray) -> np.ndarray:
        """the function to create classified score.

        Args:
            image (np.ndarray): the original image.

        Returns:
            np.ndarray: scores for each labels.
        """
        return (
            self.forward(
                image=torch.stack(
                    [self.transforms(Image.fromarray(img)) for img in images],
                    dim=0,
                ).to(self.device)
            )
            .detach()
            .cpu()
            .numpy()
        )

    def _predict_labels(self: LimeImage, images: np.ndarray) -> None:
        """predict labels.

        Args:
            images (np.ndarray): the original image.
        """
        scores: np.ndarray = self._predict(images=images)
        ranks: np.ndarray = np.argsort(scores)[:, ::-1]
        self.preds_: List[PredLabel] = list()
        self.pred_labels_: List[int] = list()
        for i, label in enumerate(ranks[0, : self.top_labels_]):
            self.preds_.append(
                PredLabel(
                    label=label,
                    name=self.labels[label],
                    rank=i,
                    score=sigmoid(scores[0, label]),
                )
            )
            self.pred_labels_.append(label)
        return

    def show_labels(self: LimeImage) -> None:
        """print predicted labels"""
        show_text("\n".join([str(pred) for pred in self.preds_]))
        return

    def draw_boundary(self: LimeImage, ax: Axes) -> None:
        """draw boundary

        Args:
            ax (Axes): the Axes instance.
        """
        draw_image_boundary(
            image=self.explain_.image,
            boundary=self.explain_.segments,
            title="Segment Boundary",
            ax=ax,
        )
        return

    def draw(
        self: LimeImage,
        ax: Axes,
        rank: Optional[int] = None,
        label: Optional[int] = None,
        draw_negative: bool = False,
        title: Optional[str] = None,
        title_model: bool = False,
        title_label: bool = False,
        title_score: bool = False,
        **kwargs: Any,
    ) -> AxesImage:
        """the main function.

        Args:
            ax (Optinonal[Axes): the Axes instance.
            rank (Optional[int]): the rank of the target class.
            label (Optional[int]): the label of the target class.
            draw_negative (bool): draw negative regions.
            title (Optional[str]): title of heatmap.
            title_model (bool): show model name in title.
            title_label (bool): show label name in title.
            title_score (bool): show score in title.

        Returns:
            AxesImage: colorbar.
        """
        if rank is None and label is None:
            rank = 0
        if rank is not None and rank >= self.top_labels_:
            raise ValueError(
                f"rank must be less than top_labels ({self.top_labels})"
            )
        if label is not None and label not in self.pred_labels_:
            raise ValueError(f"label ({label}) is not predicted")
        if label is not None:
            rank = self.pred_labels_.index(label)
        # create heatmap
        heatmap: np.ndarray = np.vectorize(
            dict(self.explain_.local_exp[self.explain_.top_labels[rank]]).get
        )(self.explain_.segments)
        # normalize heatmap
        heatmap /= heatmap.max()
        if draw_negative:
            heatmap = heatmap.clip(min=-1.0, max=1.0)
        else:
            heatmap = heatmap.clip(min=0.0, max=1.0)
        # title
        if title is None:
            label_name: str = self.labels[self.explain_.top_labels[rank]]
            if title_score:
                label_name += f" ({self.preds_[rank].score:.4f})"
            if title_model:
                title = "LIME"
                if title_label:
                    title += f" ({label_name})"
            elif title_label:
                title = label_name
        # draw the heatmap
        return draw_image_heatmap(
            image=self.explain_.image,
            heatmap=heatmap,
            ax=ax,
            title=title,
            draw_negative=draw_negative,
        )
