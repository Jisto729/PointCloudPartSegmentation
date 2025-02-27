import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os

class ShapeNetPartDataset(Dataset):
    def __init__(self, dir, split: str = 'train'):
        # split: train/val/test/train_debug/val_debug/test_debug
        self.dir = dir
        self.files = []
        self.size = 0
        debug = False

        #debug sizes: train: 800, val: 100, test: 100
        if split == 'train_debug':
            self.files = ['train0.h5']
            self.size = 800
            debug = True
        elif split == 'val_debug':
            self.files = ['val0.h5']
            self.size = 100
            debug = True
        elif split == 'test_debug':
            self.files = ['test0.h5']
            self.size = 100
            debug = True
        elif split == 'train':
            self.files = ['train0.h5','train1.h5','train2.h5','train3.h5','train4.h5', 'train5.h5']
        elif split == 'val':
            self.files = ['val0.h5']
        elif split == 'test':
            self.files = ['test0.h5','test1.h5']

        point_clouds = np.empty((0, 2048, 3))
        point_labels = np.empty((0, 2048))

        for file in self.files:
            with h5py.File(os.path.join(self.dir, file), 'r') as f:
                if not debug:
                    self.size += f["data"].shape[0]
                point_clouds = np.concatenate((point_clouds, np.array(f["data"])))
                point_labels = np.concatenate((point_labels, np.array(f["seg"])))

        self.point_clouds = torch.tensor(point_clouds[:self.size], dtype=torch.float32)
        self.point_labels = torch.tensor(point_labels[:self.size], dtype=torch.long)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        point_cloud = self.point_clouds[index]
        point_labels = self.point_labels[index]

        return point_cloud, point_labels