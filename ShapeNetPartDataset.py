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

        #debug sizes: train: 800, val: 100, test: 100
        if split == 'train_debug':
            self.files = ['train0.h5']
            self.size = 800
        elif split == 'val_debug':
            self.files = ['val0.h5']
            self.size = 100
        elif split == 'test_debug':
            self.files = ['test0.h5']
            self.size = 100
        elif split == 'train':
            self.files = ['train0.h5','train1.h5','train2.h5','train3.h5','train4.h5', 'train5.h5']
        elif split == 'val':
            self.files = ['val0.h5']
        elif split == 'test':
            self.files = ['test0.h5','test1.h5']

        if self.size == 0:
            for file in self.files:
                with h5py.File(os.path.join(self.dir, file), 'r') as f:
                    self.size += f["data"].shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index >= self.size:
            raise IndexError("Index out of range")
        #2048 = num of point clouds in each full h5 file
        file_index = index // 2048
        print(file_index)
        print(index)
        index = index % 2048
        print(self.files)
        curr_file = self.files[file_index]
        with h5py.File(os.path.join(self.dir, curr_file), 'r') as f:
            points = np.array(f["data"])
            parts = np.array(f["seg"])

        point_cloud = points[index]
        point_labels = parts[index]

        return point_cloud, point_labels