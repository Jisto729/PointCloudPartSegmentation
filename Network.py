import torch
from torch import nn
from SetAbstraction import MSGSetAbstraction, MRGSetAbstraction, SSGSetAbstraction
from FeaturePropagation import FeaturePropagation

class Network(nn.Module):
    def __init__(self, batch_size, classes, device, object_classes):
        super(Network, self).__init__()
        self.stem_MLP = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 32, 1),
            nn.ReLU(),
        )
        # self.SA_layer1 = MSGSetAbstraction(0.2, 0.4, 0.8, 32, 64, batch_size, device)
        # self.SA_layer2 = MSGSetAbstraction(0.3, 0.6, 1.0,  64, 128, batch_size, device)
        # self.SA_layer3 = MSGSetAbstraction(0.5, 1.0, 1.5, 128, 256, batch_size, device)
        # self.SA_layer4 = MSGSetAbstraction(0.8, 1.5, 2.0, 256, 512, batch_size, device)
        # self.SA_layer1 = MRGSetAbstraction(0.2, 32, 64, batch_size, device)
        # self.SA_layer2 = MRGSetAbstraction(0.4, 64, 128, batch_size, device)
        # self.SA_layer3 = MRGSetAbstraction(0.8, 128, 256, batch_size, device)
        # self.SA_layer4 = MRGSetAbstraction(1.2, 256, 512, batch_size, device)
        self.SA_layer1 = SSGSetAbstraction(0.2, 32, 64, batch_size, device)
        self.SA_layer2 = SSGSetAbstraction(0.4, 64, 128, batch_size, device)
        self.SA_layer3 = SSGSetAbstraction(0.8, 128, 256, batch_size, device)
        self.SA_layer4 = SSGSetAbstraction(1.2, 256, 512, batch_size, device)
        self.FP_layer1 = FeaturePropagation(512, 256, 256, batch_size, device)
        self.FP_layer2 = FeaturePropagation(256, 128, 128, batch_size, device)
        self.FP_layer3 = FeaturePropagation(128, 64, 64, batch_size, device)
        self.FP_layer4 = FeaturePropagation(64, 32, 32 + 3 + object_classes, batch_size, device)
        self.final_MLP = nn.Sequential(
            nn.Conv1d(32, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, classes, 1)
        )
        self.batch_size = batch_size
        self.object_classes = object_classes

    def forward(self, points, class_labels):

        #SA1 - takes: all points, outputs fourth
        #SA2 - takes fourth, outputs 16th
        #SA3 - takes 16th outputs 64th
        #SA4 - takes 64th outputs 256th

        points_features = self.stem_MLP(points)
        #SET ABSTRACTION
        SA1_centroids, SA1_centroids_features = self.SA_layer1(points, points_features.permute(0, 2, 1))
        SA2_centroids, SA2_centroids_features = self.SA_layer2(SA1_centroids, SA1_centroids_features)
        SA3_centroids, SA3_centroids_features = self.SA_layer3(SA2_centroids, SA2_centroids_features)
        SA4_centroids, SA4_centroids_features = self.SA_layer4(SA3_centroids, SA3_centroids_features)

        #FEATURE PROPAGATION
        FP_SA3_centroids_features = self.FP_layer1(SA3_centroids, SA3_centroids_features, SA4_centroids, SA4_centroids_features)
        # takes all SA2 centroids, all of which are also in SA3 it gives them the FP SA3 centroid features, and then interpolates the rest from FP SA3 centroid features
        # after that it concatenates to each SA2 centroid its SA2 centroid feature
        FP_SA2_centroids_features = self.FP_layer2(SA2_centroids, SA2_centroids_features, SA3_centroids, FP_SA3_centroids_features)
        FP_SA1_centroids_features = self.FP_layer3(SA1_centroids, SA1_centroids_features, SA2_centroids, FP_SA2_centroids_features)

        #resizing class labels so that they can be concatenated to the point features along with the 3d positions
        class_labels = class_labels.view(self.batch_size, -1, self.object_classes).repeat(1, points_features.shape[2], 1).permute(0, 2, 1)

        FP_points_features = self.FP_layer4(points, torch.cat((class_labels, points, points_features), dim=1), SA1_centroids, FP_SA1_centroids_features).contiguous()
        return self.final_MLP(FP_points_features), FP_points_features