import torch
import numpy as np
import open3d as o3d
from torch_geometric.nn import XConv, fps, global_mean_pool
from model import PointCNN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
import os.path as osp
from torch.utils.data import Dataset
import os
from tqdm import tqdm

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


# def voxel2pcd(img, threshold = 0.1):
#     B, C, H, W, D = img.shape
#     mask = img > threshold
#     img_masked = img[mask]
#     pcd = img_masked.nonzero()
#
#     return pcd




class Vessel(Dataset):
    def __init__(self, num_points = None, phase = 'train',
                 root = '/media/ymz/2b933929-0294-4162-9385-4fe3eec72189/vessel/voxel/output',
                 threshold = None, downsample =False):
        self.gaussian_noise = 0.01
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.num_points = num_points
        self.root = os.path.join(root, phase)
        self.files = []
        for _, _, item in os.walk(self.root):
            for filename in item:
                self.files.append(os.path.join(self.root, filename))
        self.threshold = threshold
        self.downsample = downsample
        print('load' + ' {} '.format(len(self.files)) + 'data')



    def __getitem__(self, item):
        data = np.load(self.files[item])
        voxel = data['voxel']
        label = data['label']
        pred = data['seg']

        if not self.threshold is None:
            seg = pred[0] > np.log(self.threshold)
        else:
            seg = pred[0] > pred[1]

        label_seg = torch.from_numpy(label.squeeze()[seg])
        pcd = self.voxel2pcd(seg)

        if self.downsample and (not self.num_points == None):
            _, mask = torch.rand(len(pcd)).topk(self.num_points)
            pcd = pcd[mask]
            label_seg = label_seg[mask]

        W, H, D = voxel[0].shape
        D = 50
        pcd = torch.mul(pcd, torch.tensor([1/W,1/H, 1/D]).reshape(1,3))

        if self.phase == 'train':
            pcd = self.jitter_pointcloud(pcd, sigma = self.gaussian_noise)

        return pcd, label_seg

    def __len__(self):
        return len(self.files)

    def jitter_pointcloud(self, pointcloud, sigma=0.01, clip=0.01):
        N, C = pointcloud.shape
        pointcloud += torch.clip(sigma * torch.randn(N, C), -1 * clip, clip)
        return pointcloud

    def voxel2pcd(self, voxel):
        x, y, z = voxel.nonzero()
        pcd = np.concatenate([x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)], axis = -1)
        return torch.from_numpy(pcd)

def collate_fn_vessel(list_data):
    pos = []
    labels = []
    batch = []
    for ind, (pcd, label) in enumerate(list_data):
        pos.append(pcd)
        batch.append(torch.ones(len(pcd)))
        labels.append(label)

    batch = torch.vstack(batch)
    if len(batch) == 1:
        pos = pcd.unsqueeze(0)
    else:
        pos = torch.vstack(pos)
    labels = torch.vstack(labels)

    return pos, batch, labels






if __name__ == "__main__":
    dataset = Vessel()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn_vessel,
        drop_last=False
    )
    for pos, batch, label in tqdm(dataloader):
        break

    #show segmentation
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pos.squeeze().numpy())
    point_cloud.paint_uniform_color([1,0,0])
    o3d.visualization.draw_geometries([point_cloud])

    #show gt
    point_cloud_seg = o3d.geometry.PointCloud()
    point_cloud_seg.points = o3d.utility.Vector3dVector(pos.squeeze()[label.squeeze() > 0].numpy())
    point_cloud_seg.paint_uniform_color([1, 0, 0])

    point_cloud_unseg = o3d.geometry.PointCloud()
    point_cloud_unseg.points = o3d.utility.Vector3dVector(pos.squeeze()[label.squeeze() < 1].numpy())
    point_cloud_unseg.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([point_cloud_seg, point_cloud_unseg])










