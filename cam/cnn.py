#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Tuple, List, Dict, DefaultDict, Callable, Optional
from abc import ABC
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.hooks import RemovableHandle
import torch.nn as nn
from torch import Tensor
from torchvision.models._api import WeightsEnum

from cam.libs_cam import TargetLayer, Activations, Gradients, Weights
from cam.libs_cam import channel_size
from cam.utils import show_table


class CNN(ABC):
    """A part of each CAM models that is responsible for the CNN model.

    CNN (Convolutional Neuarl Network) models included in
    `torchvision.models <https://pytorch.org/vision/stable/models.html>`
    consist of following three parts.

    #. "feature part": extract features from input image (net.feature)
    #. "avgpool part": average the output of the feature part (net.avgpool)
    #. "classifier part": classify image with the given labels (net.classifier)

    CAM models create a "Saliency Map" that represents
    which regions of the original image
    the CNN model pays attention (or doesn't pay attention)
    to determine the indicated label.

    To create the saliency map,
    CAM models retrieve "Activation"s
    (output of Conv. Layer when forward the original image to the CNN),
    weight them with some weight,
    and merge them over channels and layers.
    (Sometimes "Activation" is named as "Feature Map".)

    One of the strong candidates for the weight of activations is
    "Gradient"s (output of Conv. Layer when backword the score to the CNN).

    To retrieve activations and gradients,
    hook functions to target Conv. (Convolutional) Layers
    in the feature part of the CNN. (self._hook_target_layer())

    Good Conv. Layers to retrieve activation (and gradient) for
    creating "Good Saliency Map"
    (well represents which regions the CNN model pays attention or doesn't)
    are the last Conv. Layer of each block in the feature part of the CNN.
    Because the last Conv. Layer of a block of the feature part of CNN
    sums the features created in the block up.

    In general, the feature part of CNN model
    consists of stacked multiple blocks containing Conv. Layers.
    The deeper that hierarchy,
    the more channels and the fewer pixels the last Conv. Layer in the block
    (the more features and the larger region of attention).

    If some blocks (some last Conv. Layer of blocks) output the same
    channels and pixels (= shape),
    It is good to retieve activation and gradients
    only from the last block among the same shape blocks.
    (self._get_conv_layers())

    Another strong candidate for the weight of activation is
    the weight of Linear Layer in the classifier part of the CNN.
    The shape of the weight have to be (n_labels x n_channels).
    (n_labels is the number of labels,
    n_channels is the number of channels of the last Conv. Layer
    of the feature part of CNN.)

    If the classifier part of the CNN has multiple Linear Layers,
    multiply them all.
    Finally, the shape of the weight maight be (n_labels x n_channels).

    But moreover, if the last Conv. Layer of the feature part of the CNN
    outputs values larger than n_channels,
    the avgpool part of the CNN sums them up to n_channels
    (average pooling over channels).
    If so, the output size (H x W) of the AvgPool Layer is larger than 1.
    Then, we have to average the weight per the output size.
    (self._create_class_weight())

    Args:
        target (TargetLayer):
            the target layer to retrieve activation and gradient.
        net (Callable): the function creates the CNN model.
        weights (WeightsEnum): the pre-trained weights of the CNN model.
    """

    def __init__(
        self: CNN,
        target: TargetLayer,
        net: Callable,
        weights: WeightsEnum,
    ) -> None:
        # device
        self.device_: str = "cuda" if torch.cuda.is_available() else "cpu"
        # network
        self.net_weights_: WeightsEnum = weights
        self.transform_: Callable = self.net_weights_.transforms()
        self.net_: nn.Module = (
            net(weights=self.net_weights_).to(self.device_).eval()
        )
        self.softmax_: nn.Softmax = nn.Softmax(dim=1)
        # labels
        self.labels_: List[str] = self.net_weights_.meta["categories"]
        # caches
        self.init_activations_: Weights = Weights()
        self.activations_: Activations = Activations()
        self.gradients_: Gradients = Gradients()
        # hook functions to Conv. Layer(s)
        self.handles_: List[RemovableHandle] = list()
        self.conv_layers_: List[str] = list()
        self._get_conv_layers()
        self._hook_target_layer(target=target)
        # weight of the classifier block in CNN
        self.class_weights_: Tensor
        self._create_class_weights()
        return

    def _get_init_activation(
        self: CNN,
        module: nn.Module,
        actv_input: Tuple[Tensor],
        actv_output: Tensor,
    ) -> None:
        """retrieve an initial activation from a Conv. Layer when forwarding.

        initial activations are stored at self.init_activations.

        Args:
            module (nn.Module): the Layer that this function was hooked.
            actv_input (Tuple[Tensor]): forward input of the Layer.
            actv_output (Tensor): forward output of the Layer (activation).
        """
        self.init_activations_.append(weight=actv_output.clone().detach())
        return

    def _get_activation(
        self: CNN,
        module: nn.Module,
        actv_input: Tuple[Tensor],
        actv_output: Tensor,
    ) -> None:
        """retrieve an activation from a Conv. Layer when forwarding.

        activations are stored at self.activations.

        Args:
            module (nn.Module): the Layer that this function was hooked.
            actv_input (Tuple[Tensor]): forward input of the Layer.
            actv_output (Tensor): forward output of the Layer (activation).
        """
        self.activations_.append(output=actv_output)
        return

    def _get_gradient(
        self: CNN,
        module: nn.Module,
        grad_input: Tuple[Tensor],
        grad_output: Tuple[Tensor],
    ) -> None:
        """retrieve a gradient from a Conv. Layer when backwarding.

        gradients are stored at self.gradients.

        Args:
            module (nn.Module): the Layer that this function was hooked.
            grad_input (Tuple[Tensor]): backward input of the Layer.
            grad_output (Tensor): backward output of the Layer (gradient).
        """
        self.gradients_.append(output=grad_output[0])
        return

    def _hook_by_name(
        self: CNN,
        name: str,
        forward_fn: Optional[str] = None,
        backward_fn: Optional[str] = None,
    ) -> None:
        """hook functions using module name"""
        modname: str = ""
        for n in name.split("."):
            if n.isnumeric():
                modname += f"[{n}]"
            else:
                modname += f".{n}"
        cmnd: str
        if forward_fn is not None:
            cmnd = f"""self.handles_.append(
                self.net_.features{modname}.register_forward_hook(
                    {forward_fn}
                )
            )"""
            exec(cmnd)
        if backward_fn is not None:
            cmnd = f"""self.handles_.append(
                self.net_.features{modname}.register_full_backward_hook(
                    {backward_fn}
                )
            )"""
            exec(cmnd)
        return

    def _delete_handles(self: CNN) -> None:
        """remove all hooked handles"""
        for handler in self.handles_:
            handler.remove()
        self.handles_ = list()
        return

    def _get_conv_layers(self: CNN) -> None:
        """get names of the last Conv. Layers
        for each blocks in the feature part.
        """
        # clear cache
        self._delete_handles()
        self.init_activations_.clear()
        self.conv_layers_ = list()
        # get names of all Conv. Layer
        info_conv_layers: Dict[int, Dict] = dict()
        all_conv_layers: DefaultDict[List] = defaultdict(list)
        for module in self.net_.features.named_modules():
            if isinstance(module[1], nn.ReLU):
                # https://github.com/frgfm/torch-cam/issues/72
                module[1].inplace = False
                continue
            if isinstance(module[1], nn.modules.conv.Conv2d):
                block: int = int(module[0].split(".")[0])
                if block == 0:
                    # ignore the first block
                    # because the first block just expands color information
                    # to following channels
                    continue
                all_conv_layers[block].append(module[0])
        # select the last Conv. Layer for each blocks
        for block, names in all_conv_layers.items():
            last_name: str = names[-1]
            info_conv_layers[block] = {
                "block": block,
                "name": last_name,
            }
            self._hook_by_name(
                name=last_name,
                forward_fn="self._get_init_activation",
                backward_fn=None,
            )
        # create dummy (black) image
        dummy: Tensor = (
            self.transform_(
                Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
            )
            .unsqueeze(0)
            .to(self.device_)
        )
        # forward network and get gradients of hooked Conv. Layers
        with torch.no_grad():
            _ = self.net_.forward(dummy)
        assert len(self.init_activations_) == len(info_conv_layers)
        # distribute blocks according to its shape of activation
        shape_dict: DefaultDict[List] = defaultdict(list)
        for (_, shape), block in zip(
            self.init_activations_, sorted(info_conv_layers.keys())
        ):
            info_conv_layers[block]["shape"] = "x".join(
                [str(s) for s in shape]
            )
            shape_dict[shape].append(block)
        # select the last blocks over shape of activation
        last_blocks: List[int] = list()
        for shape, blocks in shape_dict.items():
            max_block: int = max(blocks)
            for block in blocks:
                is_last: bool = block == max_block
                info_conv_layers[block]["is_last"] = is_last
                if not is_last:
                    info_conv_layers[block]["offset"] = ""
            last_blocks.append(max_block)
        for i, block in enumerate(sorted(last_blocks)):
            self.conv_layers_.append(info_conv_layers[block]["name"])
            info_conv_layers[block]["offset"] = str(i)
        self.info_conv_layers_: pd.DataFrame = (
            pd.DataFrame(info_conv_layers.values())
            .sort_values(by="block", ignore_index=True)
            .loc[:, ["block", "shape", "is_last", "offset", "name"]]
        )
        # clear cache
        self._delete_handles()
        self.init_activations_.clear()
        return

    def show_conv_layers(self: CNN) -> None:
        show_table(df=self.info_conv_layers_, index=False)
        return

    def _hook_target_layer(self: CNN, target: TargetLayer) -> None:
        if target is None:
            return
        target_layers: List[str] = list()
        if isinstance(target, str):
            if target == "last":
                target_layers.append(self.conv_layers_[-1])
            elif target == "all":
                target_layers = self.conv_layers_[:]
            else:
                raise ValueError(f"target is invalid text: {target}")
        elif isinstance(target, int):
            try:
                target_layers.append(self.conv_layers_[target])
            except Exception:
                raise ValueError(f"target is invalid value: {target}")
        elif isinstance(target, list):
            if len(target) == 0:
                return
            try:
                target_layers = np.array(self.conv_layers)[target].tolist()
            except Exception:
                raise ValueError(f"target is invalid value: {target}")
        for target_layer in target_layers:
            self._hook_by_name(
                name=target_layer,
                forward_fn="self._get_activation",
                backward_fn="self._get_gradient",
            )
        return

    def _get_avgpool_size(self: CNN) -> int:
        """calc the output size of the avgpool part.

        Returns:
            int: the size of the avgpool part.
        """
        avgpool_size: int = 1
        if isinstance(self.net_.avgpool.output_size, int):
            avgpool_size = self.net_.avgpool.output_size
        else:
            for size in self.net_.avgpool.output_size:
                avgpool_size *= size
        return avgpool_size

    def _create_class_weights(self: CNN) -> None:
        """calc the weight of Linear Layer of the classifier part."""
        # if the classifier part of the CNN model has multiple Linear Layers,
        # multiply them all.
        # finally, the shape of the weight should be (n_labels x n_channels).
        # n_channels is the number of channels of the last Conv. Layer.
        # n_labels is the number of labels to predict.
        cweight: Optional[Tensor] = None
        for module in self.net_.classifier.named_modules():
            if isinstance(module[1], nn.modules.linear.Linear):
                for param in module[1].named_parameters():
                    if param[0] == "weight":
                        weight: Tensor = (
                            param[1].data.clone().detach().to(self.device_)
                        )
                        if cweight is None:
                            cweight = weight
                        else:
                            cweight = weight @ cweight
        # get the output size of the avgpool part
        avgpool_size: int = self._get_avgpool_size()
        if avgpool_size == 1:
            self.class_weights_ = cweight
            return
        # if avgpool_size > 0, the output of the feature part of the CNN model
        # is not aggregated to one value by channels of the last Conv. Layer.
        # so calc average of the weight per the kernel size of avgpool block.
        cweights: List[Tensor] = list()
        k: int = channel_size(cweight)
        assert k % avgpool_size == 0
        for i in range(k // avgpool_size):
            cweights.append(
                cweight[:, i * avgpool_size : (i + 1) * avgpool_size].mean(
                    dim=1, keepdim=True
                )
            )
        self.class_weights_ = torch.stack(cweights, dim=1).squeeze(2)
        return

    def _clear_cache(self: CNN) -> None:
        """clear all caches."""
        self.activations_.clear()
        self.gradients_.clear()
        return

    def _forward_net(self: CNN, image: Tensor) -> Tensor:
        """forward image to the whole CNN network.

        Args:
            image (Tensor): the original image.

        Returns:
            Tensor: the scores for each class. (not normalized)
        """
        return self.softmax_(self.net_.forward(image))

    def _forward_classifier(self: CNN, activation: Tensor) -> Tensor:
        """forward the activation of the last Conv. Layer
        to the classifier block (and the rest of the feature block)
        of the CNN model.

        Args:
            activation (Tensor): the activation of the last Conv. Layer.

        Returns:
            Tensor: the scores for each class. (not normalized)
        """
        return self.softmax_(self.net_.forward_classifier(activation))

    def __delete__(self: CNN) -> None:
        self._delete_handles()
        return
