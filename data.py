import torch
import numpy as np
import open3d as o3d
from torch_geometric.nn import XConv, fps, global_mean_pool
from model import PointCNN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
import os.path as osp
from torch.utils.data import Dataset


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




class Vessel(Dataset):
    def __init__(self, num_points = None, phase = 'train'):
        self.gaussian_noise = 0.01
        self.phase = phase
        self.num_points = num_points

    def __getitem__(self, item):

        if self.phase == 'train':
            pcd = jitter_pointcloud(pcd, self.gaussian_noise)

        if not self.num_points == None:
            np.random.shuffle(pcd)
            pcd = pcd[:self.num_points]
        return pcd

    def __len__(self):
        return len()

    def jitter_pointcloud(self, pointcloud, sigma=0.01, clip=0.01):
        N, C = pointcloud.shape
        pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
        return pointcloud



def collate_fn_vessel(list_data):
    pos = []
    batch = []
    for ind, pcd in enumerate(list_data):
        pos.append(pcd)
        batch.append(np.ones(len(pcd)))

    batch = np.vstack(batch)
    pos = np.array(pos)

    return torch.from_numpy(pos), torch.from_numpy(batch)






if __name__ == "__main__":
    dataset = Vessel()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn_vessel,
        drop_last=False
    )












