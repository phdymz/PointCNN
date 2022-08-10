import torch
import numpy as np
import open3d as o3d
from torch_geometric.nn import XConv, fps, global_mean_pool
from model import PointCNN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
import os.path as osp



def get_dataset(num_points):
    name = 'ModelNet10'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    pre_transform = T.NormalizeScale()
    transform = T.SamplePoints(num_points)

    train_dataset = ModelNet(
        path,
        name='10',
        train=True,
        transform=transform,
        pre_transform=pre_transform)
    test_dataset = ModelNet(
        path,
        name='10',
        train=False,
        transform=transform,
        pre_transform=pre_transform)

    return train_dataset, test_dataset


def voxel2pcd(img, threshold = 0.1):
    B, C, H, W, D = img.shape
    mask = img > threshold
    img_masked = img[mask]
    pcd = img_masked.nonzero()

    return pcd



