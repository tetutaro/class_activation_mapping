#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from cam.models.dummy_cam import DummyCAM
from cam.models.fake_cam import FakeCAM
from cam.models.vanilla_cam import VanillaCAM
from cam.models.grad_cam import GradCAM
from cam.models.grad_cam_pp import GradCAMpp
from cam.models.smooth_grad_cam_pp import SmoothGradCAMpp
from cam.models.integrad_cam import InteGradCAM
from cam.models.xgrad_cam import XGradCAM
from cam.models.hires_cam import HiResCAM
from cam.models.eigen_cam import EigenCAM
from cam.models.eigengrad_cam import EigenGradCAM
from cam.models.ablation_cam import AblationCAM
from cam.models.score_cam import ScoreCAM
from cam.models.group_cam import GroupCAM
from cam.models.cluster_cam import ClusterCAM
from cam.models.poly_cam import PolyCAM
from cam.models.cosine_cam import CosineCAM

__all__ = [
    "DummyCAM",
    "FakeCAM",
    "VanillaCAM",
    "GradCAM",
    "GradCAMpp",
    "SmoothGradCAMpp",
    "InteGradCAM",
    "XGradCAM",
    "HiResCAM",
    "EigenCAM",
    "EigenGradCAM",
    "AblationCAM",
    "ScoreCAM",
    "GroupCAM",
    "ClusterCAM",
    "PolyCAM",
    "CosineCAM",
]
