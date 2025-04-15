import torch
from torch import nn
from torch_geometric.nn import knn_interpolate

class FeaturePropagation(nn.Module):
    def __init__(self, input_channels, output_channels, skip_channels, batch_size, device):
        super(FeaturePropagation, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.skip_channels = skip_channels
        self.batch_size = batch_size
        self.device = device
        self.unit_pointNet = nn.Sequential(
            nn.Conv1d(input_channels + skip_channels, output_channels, 1),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(),
            nn.Conv1d(output_channels, output_channels, 1),
        )
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


    def forward(self, points, point_features, centroids, centroid_features,):
        #each centroid has a feature vector, each point needs a feature vector interpolated from these centroid features

        point_batch_tensor = torch.arange(self.batch_size).repeat_interleave(points.size(dim=2)).to(self.device)
        centroid_batch_tensor = torch.arange(self.batch_size).repeat_interleave(centroids.size(dim=2)).to(self.device)

        #flattening tensors, so they can be used in knn interpolate and knn functions
        points = points.reshape(-1, 3).contiguous()
        centroids = centroids.view(-1, 3).contiguous()
        centroid_features = centroid_features.view(-1, self.input_channels).contiguous()

        interpolated_features = knn_interpolate(x=centroid_features, pos_x=centroids, pos_y=points, batch_x=centroid_batch_tensor, batch_y=point_batch_tensor, k=3)
        interpolated_features = interpolated_features.view(self.batch_size, self.input_channels, -1)
        #concatenating the point features passed using the skip links from a set abstraction layer with the interpolated features from previous feature propagation layer
        concatenated_features = torch.cat((point_features, interpolated_features), dim=1).contiguous()
        #passing these features through a shared unit point net
        transformed_features = self.unit_pointNet(concatenated_features)

        return transformed_features