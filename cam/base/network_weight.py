#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Tuple, List, Dict, DefaultDict, Callable, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from torchvision.models import WeightsEnum

from cam.base import (
    DEBUG,
    Shape,
    batch_shape,
    channel_shape,
    position_shape,
    CommonWeight,
)
from cam.base.containers import Weights, Activations, Gradients
from cam.base.acquired import (
    Acquired,
    merge_acquired_list,
    merge_acquired,
)
from cam.utils.display import show_table


class NetworkWeight(CommonWeight):
    """A part of the CAM model that is responsible for the CNN model.

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
    """

    def __init__(self: NetworkWeight, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # parameters
        self.batch_size_: int = kwargs["batch_size"]
        self.n_divides_: int = kwargs["n_divides"]
        self.n_samples_: int = kwargs["n_samples"]
        self.sigma_: float = kwargs["sigma"]
        # network
        self.net_weights_: WeightsEnum = kwargs["weights"]
        self.net_: nn.Module = (
            kwargs["net"](weights=self.net_weights_).to(self.device).eval()
        )
        self.softmax_: nn.Softmax = nn.Softmax(dim=1)
        # transform function from PIL.Image to Tensor
        self.transforms: Callable[
            [Image], Tensor
        ] = self.net_weights_.transforms()
        # labels
        self.labels: List[str] = self.net_weights_.meta["categories"]
        self.n_labels: int = len(self.labels)
        # gray image
        self.gray_: Tensor = (
            self.transforms(
                Image.fromarray(np.ones((256, 256, 3), dtype=np.uint8))
            )
            .unsqueeze(0)
            .to(self.device)
            * 128
        )
        # caches
        self.target_layers_: Optional[List[str]] = None
        self.handles_: List[RemovableHandle] = list()
        self.init_activations_: Optional[Weights] = None
        self.activations_: Optional[Activations] = None
        self.gradients_: Optional[Gradients] = None
        # information of Conv. Layers
        self.conv_layers: List[str]
        self.n_channels_last_: int
        self.info_conv_layers_: pd.DataFrame
        # get information of Conv. Layers
        self._get_conv_layers()
        return

    # ## hook functions

    def _get_init_activation(
        self: NetworkWeight,
        module: nn.Module,
        actv_input: Tuple[Tensor],
        actv_output: Tensor,
    ) -> None:
        """retrieve an initial activation from a Conv. Layer when forwarding.

        Args:
            module (nn.Module): the Layer that this function was hooked.
            actv_input (Tuple[Tensor]): forward input of the Layer.
            actv_output (Tensor): forward output of the Layer (activation).
        """
        self.init_activations_.append(weight=actv_output.clone().detach())
        return

    def _get_activation(
        self: NetworkWeight,
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
        self.activations_.append(activation=actv_output.clone().detach())
        return

    def _get_gradient(
        self: NetworkWeight,
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
        self.gradients_.append(gradient=grad_output[0].clone().detach())
        return

    def _hook_by_name(
        self: NetworkWeight,
        name: str,
        forward_fn: str,
        backward_fn: Optional[str],
    ) -> None:
        """register hook functions by the module name

        Args:
            name (str): the name of the target layer.
            forward_fn (str): the name of forward hook function.
            backward_fn (Optional[str]): the name of backward hook function.
        """
        modname: str = ""
        for n in name.split("."):
            if n.isnumeric():
                modname += f"[{n}]"
            else:
                modname += f".{n}"
        cmnd: str
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

    def _clear_cache(self: NetworkWeight) -> None:
        """clear all caches."""
        if self.init_activations_ is not None:
            self.init_activations_.clear()
            self.init_activations_ = None
        if self.activations_ is not None:
            self.activations_.clear()
            self.activations_ = None
        if self.gradients_ is not None:
            self.gradients_.clear()
            self.gradients_ = None
        while True:
            try:
                handle: RemovableHandle = self.handles_.pop()
                handle.remove()
            except IndexError:
                break
        self.target_layers_ = None
        return

    def __delete__(self: NetworkWeight) -> None:
        """delete all caches."""
        self._clear_cache()
        return

    # ## fuctions for Conv. Layers information

    def _get_conv_layers(self: NetworkWeight) -> None:
        """get names of the last Conv. Layers
        for each blocks in the feature part.
        """
        # clear cache
        self._clear_cache()
        self.init_activations_ = Weights()
        self.conv_layers = list()
        # get names of all Conv. Layer
        info_conv_layers: Dict[int, Dict] = dict()
        all_conv_layers: DefaultDict[List] = defaultdict(list)
        for module in self.net_.features.named_modules():
            if isinstance(module[1], nn.ReLU):
                # https://github.com/frgfm/torch-cam/issues/72
                module[1].inplace = False
                continue
            if isinstance(module[1], nn.Conv2d):
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
        # forward network and get activations of hooked Conv. Layers
        with torch.no_grad():
            _ = self.net_.forward(self.gray_)
        # number of channels of last Conv. Layer
        assert len(self.init_activations_) == len(info_conv_layers)
        self.n_channels_last_ = channel_shape(self.init_activations_[-1])
        # distribute blocks according to its position shape of activation
        shape_dict: DefaultDict[List] = defaultdict(list)
        for activation, block in zip(
            self.init_activations_, sorted(info_conv_layers.keys())
        ):
            info_conv_layers[block]["shape"] = "x".join(
                [str(s) for s in activation.shape]
            )
            shape_dict[position_shape(activation)].append(block)
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
            self.conv_layers.append(info_conv_layers[block]["name"])
            info_conv_layers[block]["offset"] = str(i)
        # create pandas.DataFrame
        self.info_conv_layers_ = (
            pd.DataFrame(info_conv_layers.values())
            .sort_values(by="block", ignore_index=True)
            .loc[:, ["block", "shape", "is_last", "offset", "name"]]
        )
        # clear cache
        self._clear_cache()
        return

    def show_conv_layers(self: NetworkWeight) -> None:
        """show information of Conv. Layers"""
        show_table(df=self.info_conv_layers_, index=False)
        return

    # ## register function for target Conv. Layer(s)

    def _register_target(
        self: NetworkWeight, target_layers: List[str]
    ) -> None:
        """hook functions to target Conv. Layers.

        Args:
            target_list (List[int]): the target to hook functions.
        """
        if len(self.conv_layers) == 0:
            raise SystemError("no Conv. Layers detected")
        if len(target_layers) == 0:
            raise SystemError("no target indicated")
        # clear cache
        self._clear_cache()
        # register hook functions
        self.activations_ = Activations()
        self.gradients_ = Gradients()
        for target_layer in target_layers:
            self._hook_by_name(
                name=target_layer,
                forward_fn="self._get_activation",
                backward_fn="self._get_gradient",
            )
        self.target_layers_ = target_layers
        return

    # ## functions about weights of the classifier part of CNN

    def _get_avgpool_size(self: NetworkWeight) -> int:
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

    def get_class_weights(self: NetworkWeight) -> Tensor:
        """calc the weight of Linear Layer of the classifier part.

        Returns:
            Tensor: the weight of Linear Layer of the classifier part.
        """
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
                            param[1].data.clone().detach().to(self.device)
                        )
                        if cweight is None:
                            cweight = weight
                        else:
                            cweight = weight @ cweight
        # get the output size of the avgpool part
        avgpool_size: int = self._get_avgpool_size()
        if avgpool_size == 1:
            if DEBUG:
                assert batch_shape(cweight) == self.n_labels
                assert channel_shape(cweight) == self.n_channels_last_
            return cweight
        # if avgpool_size > 0, the output of the feature part of the CNN model
        # is not aggregated to one value by channels of the last Conv. Layer.
        # so calc average of the weight per the kernel size of avgpool block.
        cweight_list: List[Tensor] = list()
        k: int = channel_shape(cweight)
        if DEBUG:
            assert k % avgpool_size == 0
        for i in range(k // avgpool_size):
            cweight_list.append(
                cweight[:, i * avgpool_size : (i + 1) * avgpool_size].mean(
                    dim=1, keepdim=True
                )
            )
        class_weight: Tensor = torch.stack(cweight_list, dim=1).squeeze(2)
        if DEBUG:
            assert batch_shape(class_weight) == self.n_labels_
            assert channel_shape(class_weight) == self.n_channels_last_
        return class_weight

    # ## forward functions

    def _forward_batches(self: NetworkWeight, image: Tensor) -> Tensor:
        """forward image to CNN for each batches.

        Args:
            image (Tensor): the image.

        Returns:
            Tensor: scores for each labels
        """
        # forward network
        with torch.no_grad():
            scores: Tensor = self.softmax_(self.net_.forward(image))
        return scores.clone().detach()

    def forward(self: NetworkWeight, image: Tensor) -> Tensor:
        """forward image to CNN.

        Args:
            image (Tensor): the image.

        Returns:
            Tensor: scores for each labels
        """
        if DEBUG:
            assert self.target_layers_ is None
        # calc number of batches
        b: int = batch_shape(image)
        n_batches: int = b // self.batch_size_
        if b % self.batch_size_ > 0:
            n_batches += 1
        # forward image for each batch
        scores_list: List[Tensor] = list()
        for i in range(n_batches):
            begin: int = i * self.batch_size_
            end: int = min((i + 1) * self.batch_size_, b)
            scores_list.append(
                self._forward_batches(
                    image=image[begin:end, ...],
                )
            )
        scores: Tensor = torch.cat(scores_list, dim=0)
        if DEBUG:
            assert batch_shape(scores) == b
            assert channel_shape(scores) == self.n_labels
        return scores

    # ## acquire functions

    def _simple_grad(
        self: NetworkWeight,
        image: Tensor,
        label: int,
        smooth: str,
    ) -> Acquired:
        """create batch for self._grad_batches().

        Args:
            image (Tensor): the image.
            label (int): the target label.
            smooth (str): the method of smoothing gradients.

        Returns:
            Acquired: the result of forwarding and backwarding the image.
        """
        # calc number of batches
        b: int = batch_shape(image)
        # forward image for each batch
        acquired_list: List[Acquired] = list()
        for i in range(b):
            # forward network
            logit: Tensor = self.softmax_(self.net_.forward(image[[i], ...]))
            # get scores
            scores: Tensor = logit.clone().detach()
            # backward network
            self.net_.zero_grad()
            logit[0, label].backward(retain_graph=False)
            # create aquired
            self.activations_.finalize()
            self.gradients_.finalize()
            acquired: Acquired = Acquired(
                activations=self.activations_.clone(),
                gradients=self.gradients_.clone(),
                scores=scores,
            )
            acquired.finalize()
            # clear cache
            self.activations_.clear()
            self.gradients_.clear()
            # add acquired to the list
            acquired_list.append(acquired)
        return merge_acquired_list(acquired_list=acquired_list)

    def _integrate_grad(
        self: NetworkWeight,
        image: Tensor,
        label: int,
        smooth: str,
    ) -> Acquired:
        """IntegratedGrads

        M. Sundararajan, et al.
        "Axiomatic Attribution for Deep Networks"
        ICML 2017.

        https://arxiv.org/abs/1703.01365

        Args:
            image (Tensor): the image.
            label (int): the target label.
            smooth (str): the method of smoothing gradients.

        Returns:
            Acquired: the result of forwarding and backwarding the image.
        """
        # create center and delta
        center: Tensor = self.gray_.clone()
        delta: Tensor = image - center
        shape: Shape = image.size()
        # create path from gray to image
        path_list: List[Tensor] = list()
        for i in range(1, self.n_divides_ + 1):
            path: Tensor = (i * delta / self.n_divides_) + center
            path_list.append(path)
        pathes: Tensor = torch.stack(path_list, dim=1).view(-1, *shape[1:])
        # forward and integrate
        acquired: Acquired = self._simple_grad(
            image=pathes,
            label=label,
            smooth=smooth,
        )
        return merge_acquired(
            acquired=acquired,
            unit_size=self.n_divides_,
            merge_type="integral",
        )

    def _smooth_grad(
        self: NetworkWeight,
        image: Tensor,
        label: int,
        smooth: str,
    ) -> Acquired:
        """SmoothGrad

        D. Smilov, et al.
        "SmoothGrad: removing noise by adding noise"
        arXiv 2017.

        https://arxiv.org/abs/1706.03825

        Args:
            image (Tensor): the image.
            label (int): the target label.
            smooth (str): the method of smoothing gradients.

        Returns:
            Acquired: the result of forwarding and backwarding the image.
        """
        fn: Callable[[Tensor, int, str], Acquired]
        if "integral" in smooth:
            fn = self._integrate_grad
        else:
            fn = self._simple_grad
        # create noised images
        shape: Shape = image.size()
        noised_list: List[Tensor] = list()
        for _ in range(self.n_samples_):
            noised: Tensor = torch.normal(mean=image, std=self.sigma_).to(
                self.device
            )
            noised_list.append(noised)
        noiseds: Tensor = torch.stack(noised_list, dim=1).view(-1, *shape[1:])
        # forward and smooth
        acquired: Acquired = fn(
            image=noiseds,
            label=label,
            smooth=smooth,
        )
        return merge_acquired(
            acquired=acquired,
            unit_size=self.n_samples_,
            merge_type="smooth",
        )

    def acquires_grad(
        self: NetworkWeight,
        target_layers: List[str],
        image: Tensor,
        label: int,
        smooth: str,
    ) -> Acquired:
        """forward and backward image to CNN and get activations and gradients.

        Args:
            target_layers (List[str]): the target layer(s).
            image (Tensor): the image.
            label (int): the target label.
            smooth (str): the method of smoothing gradients.

        Returns:
            Acquired: the result of forwarding and backwarding the image.
        """
        self._register_target(target_layers=target_layers)
        fn: Callable[[Tensor, int, str], Acquired]
        if "noise" in smooth:
            fn = self._smooth_grad
        elif "integral" in smooth:
            fn = self._integrate_grad
        else:
            fn = self._simple_grad
        acquired: Acquired = fn(
            image=image,
            label=label,
            smooth=smooth,
        )
        if DEBUG:
            assert len(acquired.activations) == len(self.target_layers_)
            assert batch_shape(acquired.scores) == batch_shape(image)
            assert channel_shape(acquired.scores) == self.n_labels
        self._clear_cache()
        return acquired

    # ## classify functions

    def _classify_batches(
        self: NetworkWeight,
        activation: Tensor,
    ) -> Tensor:
        """forward activation to the classifier part of CNN for each batches.

        Args:
            activation (Tensor): activation of the last Conv. Layer

        Returns:
            Tensor: scores
        """
        with torch.no_grad():
            scores: Tensor = self.softmax_(
                self.net_.forward_classifier(activation=activation)
            )
        return scores.clone().detach()

    def classify(
        self: NetworkWeight,
        activation: Tensor,
    ) -> Tensor:
        """forward activation to the classifier part of CNN.

        Args:
            activation (Tensor): activation of the last Conv. Layer

        Returns:
            Tensor: scores
        """
        if DEBUG:
            assert self.target_layers_ is None
        b: int = batch_shape(activation)
        n_batches: int = b // self.batch_size_
        if b % self.batch_size_ > 0:
            n_batches += 1
        scores_list: List[Tensor] = list()
        for i in range(n_batches):
            begin: int = i * self.batch_size_
            end: int = min((i + 1) * self.batch_size_, b)
            scores_list.append(
                self._classify_batches(
                    activation=activation[begin:end, ...],
                )
            )
        scores: Tensor = torch.cat(scores_list, dim=0)
        if DEBUG:
            assert batch_shape(scores) == b
            assert channel_shape(scores) == self.n_labels
        return scores
