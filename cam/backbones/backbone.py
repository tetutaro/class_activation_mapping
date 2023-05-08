#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from typing import Callable, TypedDict

from torchvision.models import WeightsEnum


class Backbone(TypedDict):
    """the resource container class of CNN model.

    Args:
        net (Callable): the function creates the CNN module.
        weights (WeightsEnum): the pre-trained weights of the CNN model.
    """

    net: Callable
    weights: WeightsEnum
