import os
import matplotlib.pyplot as plt
from collections import Counter
import open3d as o3d
import json
from torch.utils.data import Dataset


class BuildingNetDataset(Dataset):
    def __init__(self, point_cloud_dir, label_dir):
        self.point_cloud_dir = point_cloud_dir
        self.label_dir = label_dir
        self.label_files = sorted(os.listdir(label_dir))
        self.size = len(os.listdir(self.label_dir))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        with open(os.path.join(self.label_dir, self.label_files[index]), 'r') as file:
            labels = json.load(file)

        label_file_suffix = "_label.json"
        point_cloud_file = self.label_files[index][:-len(label_file_suffix)] + ".ply"
        point_cloud = o3d.io.read_point_cloud(os.path.join(self.point_cloud_dir, point_cloud_file))

        return point_cloud, labels

    def get_model_labels(self, model_index):
        with open(os.path.join(self.label_dir, self.label_files[model_index]), 'r') as file:
            labels = json.load(file)
        return labels


class DatasetVisualizer:
    def __init__(self, dataset: BuildingNetDataset):
        self.dataset: BuildingNetDataset = dataset
        self.labeled_points_histogram: Counter
        self.labels_per_model_histogram: Counter

        self.labeled_points_histogram, self.labels_per_model_histogram = self.__create_histograms()


    def __create_histograms(self):
        labeled_points_histogram = Counter()
        labels_per_model_histogram = Counter()

        for i in range(self.dataset.size):
            model_histogram = self.__get_model_label_histogram(i)

            labeled_points_histogram.update(model_histogram)
            for label in model_histogram:
                if label not in labels_per_model_histogram:
                    labels_per_model_histogram[label] = 1
                else:
                    labels_per_model_histogram[label] += 1
        return labeled_points_histogram, labels_per_model_histogram


    def __get_model_label_histogram(self, model_index):
        model_labels = self.dataset.get_model_labels(model_index)
        return Counter(model_labels.values())


    def display_histogram_table(self):
        fig, ax = plt.subplots()
        fig.set_figheight(len(self.labeled_points_histogram.keys()) * 0.3)
        fig.set_figwidth(6)

        ax.axis('off')
        ax.axis('tight')
        label_ids = list(set(self.labeled_points_histogram.keys()).union(self.labels_per_model_histogram.keys()))  # Combine keys
        num_of_labeled_points = list(self.labeled_points_histogram.get(label_id, 0) for label_id in label_ids)
        num_of_labels_per_model = list(self.labels_per_model_histogram.get(label_id, 0) for label_id in label_ids)

        table_data = []
        for i in range(len(label_ids)):
            table_data.append([label_ids[i], num_of_labeled_points[i], num_of_labels_per_model[i]])

        table = ax.table(cellText=table_data,#list(sorted(self.labeled_points_histogram.items())),
                            colLabels=["Label ID", "Labeled Points", "Number of Models with Label"],
                            loc='center', cellLoc='center', edges='horizontal')

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.4)
        table.auto_set_column_width(col=list(range(len(table_data[0]))))

        plt.show()


    def display_histogram_chart(self):
        fig, ax = plt.subplots()

        ax.set_xscale('log')
        ax.set_xlabel('Number of Labeled Points')
        ax.set_ylabel('Label ID')
        ax.invert_yaxis()
        ax.barh(self.labeled_points_histogram.keys(), self.labeled_points_histogram.values())

        plt.show()