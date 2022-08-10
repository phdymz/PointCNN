import torch
import numpy as np
import open3d as o3d
from torch_geometric.nn import XConv, fps, global_mean_pool
from model import PointCNN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from data import get_dataset
from torch_geometric.loader import DataLoader



if __name__ == "__main__":
    train_dataset, test_dataset = get_dataset(2048)
    batch_size = 1
    train_loader = DataLoader(test_dataset, batch_size, shuffle=True)
    model = PointCNN().cuda()


    for data in train_loader:
        data = data.cuda()
        print(data.pos.shape, data.batch.shape)
        output = model(data.pos, data.batch)
        break

    print('finish')
