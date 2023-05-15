#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from typing import Callable, TypedDict

from torchvision.models import WeightsEnum


class Backbone(TypedDict):
    """the resource container class of CNN model.

    Args:
        cnn_name (str): the name of the backbone CNN model.
        net (Callable): the function creates the CNN model.
        weights (WeightsEnum): the pre-trained weights of the CNN model.
    """

    cnn_name: str
    net: Callable
    weights: WeightsEnum
