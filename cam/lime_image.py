#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Optional, NamedTuple

import numpy as np
from PIL import Image
from lime.lime_image import LimeImageExplainer, ImageExplanation
import torch
import matplotlib as mpl

from cam.cnn import CNN
from cam.libs_cam import ResourceCNN
from cam.utils import show_text, draw_image_boundary, draw_image_heatmap


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


class LimeImage(CNN):
    def __init__(
        self: LimeImage,
        resource: ResourceCNN,
        path: str,
        random_state: int,
        top_labels: int = 5,
        num_features: int = 100000,
        num_samples: int = 1000,
    ) -> None:
        """how to use the LimeImageExplainer.

        Args:
            resource (ResourceCNN): resource of the CNN model.
            path (str): the pathname of the original image.
            ramdom_state (int): the random seed.
            top_labels (int): number of top labels to predict.
            num_features (int): number of features.
            num_samples (int): number of samplings.
        """
        super().__init__(**resource, target=None)
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
            labels=self.labels_,
            top_labels=self.top_labels_,
            num_features=num_features,
            num_samples=num_samples,
            hide_color=0,
            random_seed=random_state,
        )
        return

    def _predict(self: LimeImage, images: np.ndarray) -> np.ndarray:
        """the function to create classified score.

        Args:
            image (np.ndarray): the original image.

        Returns:
            np.ndarray: scores for each labels.
        """
        return (
            self.net_.forward(
                torch.stack(
                    [self.transform_(Image.fromarray(img)) for img in images],
                    dim=0,
                ).to(self.device_)
            )
            .softmax(dim=1)
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
                    name=self.labels_[label],
                    rank=i,
                    score=scores[0, label],
                )
            )
            self.pred_labels_.append(label)
        return

    def show_labels(self: LimeImage) -> None:
        """print predicted labels"""
        show_text("\n".join([str(pred) for pred in self.preds_]))
        return

    def draw_boundary(self: LimeImage) -> None:
        """draw boundary"""
        draw_image_boundary(
            image=self.explain_.image,
            boundary=self.explain_.segments,
            title="Segment Boundary",
        )
        return

    def draw(
        self: LimeImage,
        rank: Optional[int] = None,
        label: Optional[int] = None,
        draw_negative: bool = False,
        fig: Optional[mpl.figure.Figure] = None,
        ax: Optional[mpl.axes.Axes] = None,
    ) -> None:
        """the main function.

        Args:
            rank (Optional[int]): the rank of the target class.
            label (Optional[int]): the label of the target class.
            draw_negative (bool): draw negative regions.
            fig (Optional[mpl.figure.Figure]):
                the Figure instance that the output image is drawn.
            ax (Optinonal[mpl.axies.Axes):
                the Axes instance that the output image is drawn.
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
        # create title
        title: str = "LIME"
        if fig is None or ax is None:
            title += f" ({self.labels_[self.explain_.top_labels[rank]]})"
        # draw
        draw_image_heatmap(
            image=self.explain_.image,
            heatmap=heatmap,
            title=title,
            draw_negative=draw_negative,
            fig=fig,
            ax=ax,
        )
        return
