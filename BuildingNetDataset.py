import os
import matplotlib.pyplot as plt
from collections import Counter
import open3d as o3d
import json
from torch.utils.data import Dataset


class BuildingNetDataset(Dataset):
    def __init__(self, point_cloud_dir, label_dir, splits_dir, split: str = 'train'):
        #split: train/val/test/all
        self.point_cloud_dir = point_cloud_dir
        self.label_dir = label_dir
        self.splits_dir = splits_dir
        if split == 'train':
            model_names_file = 'train_split.txt'
        elif split == 'val':
            model_names_file = 'val_split.txt'
        elif split == 'test':
            model_names_file = 'test_split.txt'
        else:
            model_names_file = 'dataset_models.txt'
        with open(os.path.join(splits_dir, model_names_file), 'r') as f:
            self.model_names = sorted(f.read().split('\n'))
        self.size = len(self.model_names)
        self.split = split

    def __len__(self):
        return self.size

    #TODO add loading into init
    def __getitem__(self, index):
        with open(os.path.join(self.label_dir, self.model_names[index] + '_label.json'), 'r') as file:
            labels = json.load(file)

        point_cloud_file = self.model_names[index] + ".ply"
        point_cloud = o3d.io.read_point_cloud(os.path.join(self.point_cloud_dir, point_cloud_file))

        return point_cloud, labels

    def get_model_labels(self, model_index):
        with open(os.path.join(self.label_dir, self.model_names[model_index] + '_label.json'), 'r') as file:
            labels = json.load(file)
        return labels


class DatasetVisualizer:
    def __init__(self, dataset: BuildingNetDataset):
        self.dataset: BuildingNetDataset = dataset

        #number of labeled points per label
        self.labeled_points_histogram: Counter
        #number of models, where each label is used at least once
        self.models_per_label_histogram: Counter

        self.labeled_points_histogram, self.models_per_label_histogram = self.__create_histograms()
        if dataset.split == 'train':
            self.split = 'Training Split'
        elif dataset.split == 'val':
            self.split = 'Validation Split'
        elif dataset.split == 'test':
            self.split = 'Test Split'
        else:
            self.split = 'Whole Dataset'

    def __create_histograms(self):
        labeled_points_histogram = Counter()
        models_per_label_histogram = Counter()

        for i in range(self.dataset.size):
            model_histogram = self.__get_model_label_histogram(i)

            labeled_points_histogram.update(model_histogram)
            for label in model_histogram:
                if label not in models_per_label_histogram:
                    models_per_label_histogram[label] = 1
                else:
                    models_per_label_histogram[label] += 1
        return labeled_points_histogram, models_per_label_histogram


    def __get_model_label_histogram(self, model_index):
        model_labels = self.dataset.get_model_labels(model_index)
        return Counter(model_labels.values())


    def display_histogram_table(self):
        fig, ax = plt.subplots()
        fig.set_figheight(len(self.labeled_points_histogram.keys()) * 0.3)
        fig.set_figwidth(6)

        ax.axis('off')
        ax.axis('tight')

        label_ids = list(set(self.labeled_points_histogram.keys()).union(self.models_per_label_histogram.keys()))  # Combine keys
        num_of_labeled_points = list(self.labeled_points_histogram.get(label_id, 0) for label_id in label_ids)
        num_of_labels_per_model = list(self.models_per_label_histogram.get(label_id, 0) for label_id in label_ids)

        table_data = []
        for i in range(len(label_ids)):
            table_data.append([label_ids[i], num_of_labeled_points[i], num_of_labels_per_model[i]])

        table = ax.table(cellText=table_data,
                            colLabels=["Label ID", "Labeled Points", "Number of Models with Label"],
                            loc='center', cellLoc='center', edges='horizontal')

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.4)
        table.auto_set_column_width(col=list(range(len(table_data[0]))))
        ax.set_title(self.split + " Labeled Points and Model Distribution", pad=25, fontweight='bold', fontsize=14)

        plt.show()


    def display_histogram_chart(self):
        fig, ax = plt.subplots()

        ax.set_title(self.split + " Labeled Points per Label", pad=10, fontweight='bold', fontsize=14)
        ax.set_xscale('log')
        ax.set_xlabel('Number of Labeled Points')
        ax.set_ylabel('Label ID')
        ax.invert_yaxis()
        ax.barh(self.labeled_points_histogram.keys(), self.labeled_points_histogram.values())

        plt.show()