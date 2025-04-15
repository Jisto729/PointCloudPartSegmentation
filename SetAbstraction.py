import torch
from torch import nn
from NetworkUtils import sample, group, groupMRG

class SSGSetAbstraction(nn.Module):
    def __init__(self, ball_query_radius, input_channels, output_channels, batch_size, device,
                 max_group_size=1024) -> None:
        super(SSGSetAbstraction, self).__init__()
        self.ball_query_radius = ball_query_radius
        self.max_group_size = max_group_size
        self.batch_size = batch_size
        self.device = device
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.point_net = nn.Sequential(
            nn.Conv2d(input_channels + 3, input_channels, 1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            nn.Conv2d(input_channels, input_channels, 1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            nn.Conv2d(input_channels, output_channels, 1),
        )
        self.init_weights()

    #TODO redo
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, points: torch.tensor, point_features: torch.tensor,):
        """
        Layer forward method
        Args:
            points: point tensor of shape [batch_size, 3, num_points]
            point_features: point feature tensor of shape [batch_size, input_feature_dim, num_points]

        Returns: centroids of shape [batch_size, 3, num_points/4],
            centroid features of shape: [batch_size, output_feature_dim, num_points/4]

        """

        point_batch_tensor = torch.arange(self.batch_size).repeat_interleave(points.size(dim=2)).to(self.device)
        centroid_batch_tensor = torch.arange(self.batch_size).repeat_interleave(points.size(dim=2) // 4).to(self.device)

        #sampling
        points = points.permute(0, 2, 1)
        #flattening, so the tensor can be used in the fps function
        points = points.reshape(-1, 3).contiguous()
        indices = sample(points, point_batch_tensor, self.batch_size)
        centroids = points[indices]

        #grouping
        point_features = point_features.permute(0, 2, 1)
        point_features = point_features.reshape(-1, self.input_channels).contiguous()
        regions = group(self.ball_query_radius, centroids, points, point_features, point_batch_tensor,
                        centroid_batch_tensor, self.max_group_size, self.batch_size)

        #pointNet
        #reshaping regions tensor, so that it can be processed by the conv2d layers in pointnet
        regions = regions.permute(0, 3, 2, 1).contiguous()
        centroid_features = self.pointNet(regions)

        #unflattening the centroids, because every layer expects points with batch as a separate dimension
        centroids = centroids.view(self.batch_size, 3, -1)
        return centroids, centroid_features

    def pointNet(self, grouped_points):
        point_features = self.point_net(grouped_points)
        #max pooling
        global_features = torch.max(point_features, dim=2)[0]
        return global_features


class MSGSetAbstraction(nn.Module):
    def __init__(self, ball_query_radius_1, ball_query_radius_2, ball_query_radius_3, input_channels, output_channels, batch_size, device,
                 max_group_size=1024) -> None:
        super(MSGSetAbstraction, self).__init__()
        self.ball_query_radius_1 = ball_query_radius_1
        self.ball_query_radius_2 = ball_query_radius_2
        self.ball_query_radius_3 = ball_query_radius_3
        self.max_group_size = max_group_size
        self.batch_size = batch_size
        self.device = device
        self.input_channels = input_channels
        self.scales_output_channels = [output_channels // 2, output_channels // 4, output_channels // 4]
        self.point_nets = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels + 3, input_channels, 1),
                nn.ReLU(),
                nn.Conv2d(input_channels, input_channels, 1),
                nn.ReLU(),
                nn.Conv2d(input_channels, self.scales_output_channels[i], 1),
            ) for i in range(3)
        ])
        self.init_weights()

    # TODO redo
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, points: torch.tensor, point_features: torch.tensor,):
        """
        Layer forward method
        Args:
            points: point tensor of shape [batch_size, 3, num_points]
            point_features: point feature tensor of shape [batch_size, input_feature_dim, num_points]

        Returns: centroids of shape [batch_size, 3, num_points/4],
            centroid features of shape: [batch_size, output_feature_dim, num_points/4]

        """
        point_batch_tensor = torch.arange(self.batch_size).repeat_interleave(points.size(dim=2)).to(self.device)
        centroid_batch_tensor = torch.arange(self.batch_size).repeat_interleave(points.size(dim=2) // 4).to(self.device)

        #sampling
        points = points.permute(0, 2, 1)
        #flattening, so the tensor can be used in the fps function
        points = points.reshape(-1, 3)
        indices = sample(points, point_batch_tensor, self.batch_size)
        centroids = points[indices]

        #grouping
        point_features = point_features.permute(0, 2, 1)
        point_features = point_features.reshape(-1, self.input_channels)
        #Multiscale grouping
        small_regions = group(self.ball_query_radius_1, centroids, points, point_features, point_batch_tensor,
                                   centroid_batch_tensor, self.max_group_size, self.batch_size)
        medium_regions = group(self.ball_query_radius_2, centroids, points, point_features, point_batch_tensor,
                                    centroid_batch_tensor, self.max_group_size, self.batch_size)
        large_regions = group(self.ball_query_radius_3, centroids, points, point_features, point_batch_tensor,
                                   centroid_batch_tensor, self.max_group_size, self.batch_size)

        #pointNet
        #reshaping regions tensor, so that it can be processed by the conv2d layers in pointnet
        small_regions = small_regions.permute(0, 3, 2, 1).contiguous()
        medium_regions = medium_regions.permute(0, 3, 2, 1).contiguous()
        large_regions = large_regions.permute(0, 3, 2, 1).contiguous()
        # small_regions = small_regions.view(-1, self.input_channels + 3, self.max_group_size)
        # medium_regions = medium_regions.view(-1, self.input_channels + 3, self.max_group_size)
        # large_regions =  large_regions.view(-1, self.input_channels + 3, self.max_group_size)
#         print(small_regions.shape)
        small_centroid_features = self.pointNet(small_regions, 0)
        medium_centroid_features = self.pointNet(medium_regions, 1)
        large_centroid_features = self.pointNet(large_regions, 2)

        centroid_features = torch.cat([small_centroid_features.view(self.batch_size, self.scales_output_channels[0], -1),
                                       medium_centroid_features.view(self.batch_size, self.scales_output_channels[1], -1),
                                       large_centroid_features.view(self.batch_size, self.scales_output_channels[2], -1)], dim=1)

        #unflattening the centroids, because every layer expects points with batch as a separate dimension
        centroids = centroids.view(self.batch_size, 3, -1)
        return centroids, centroid_features

    def pointNet(self, grouped_points, size):
        point_features = self.point_nets[size](grouped_points)
        global_features = torch.max(point_features, dim=2)[0]
        return global_features


class MRGSetAbstraction(nn.Module):
    def __init__(self, ball_query_radius, input_channels, output_channels, batch_size, device,
                 max_group_size=1024) -> None:
        super(MRGSetAbstraction, self).__init__()
        self.ball_query_radius = ball_query_radius
        self.max_group_size = max_group_size
        self.batch_size = batch_size
        self.device = device
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.point_point_net =  nn.Sequential(
            nn.Conv2d(3, input_channels, 1),
            nn.ReLU(),
            nn.Conv2d(input_channels, input_channels, 1),
            nn.ReLU(),
            nn.Conv2d(input_channels, output_channels // 2, 1)
        )
        self.feature_point_net = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 1),
            nn.ReLU(),
            nn.Conv2d(input_channels, input_channels, 1),
            nn.ReLU(),
            nn.Conv2d(input_channels, output_channels // 2, 1)
        )

        self.init_weights()

    def init_weights(self):
        """Apply Xavier initialization to all Linear layers"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    #input: points: [B,3,N], point_features: [B,F,N]
    #output: centroids: [B,3,N/4], centroid_features: [B,F2,N/4]
    def forward(self, points: torch.tensor, point_features: torch.tensor,):
        """
        Layer forward method
        Args:
            points: point tensor of shape [batch_size, 3, num_points]
            point_features: point feature tensor of shape [batch_size, input_feature_dim, num_points]

        Returns: centroids of shape [batch_size, 3, num_points/4],
            centroid features of shape: [batch_size, output_feature_dim, num_points/4]

        """
        point_batch_tensor = torch.arange(self.batch_size).repeat_interleave(points.size(dim=2)).to(self.device)
        centroid_batch_tensor = torch.arange(self.batch_size).repeat_interleave(points.size(dim=2) // 4).to(self.device)

        #sampling
        points = points.permute(0, 2, 1)
        #flattening, so the tensor can be used in the fps function
        points = points.reshape(-1, 3).contiguous()
        indices = sample(points, point_batch_tensor, self.batch_size)
        centroids = points[indices]

        #grouping
        point_features = point_features.permute(0, 2, 1)
        point_features = point_features.reshape(-1, self.input_channels)

        #Multi resolution grouping
        region_points, region_features = groupMRG(self.ball_query_radius, centroids, points, point_features,
                                                  point_batch_tensor, centroid_batch_tensor, self.max_group_size,
                                                  self.batch_size)

        #pointNet
        #reshaping regions tensor, so that it can be processed by the conv2d layers in pointnet

        region_points = region_points.permute(0, 3, 2, 1).contiguous()
        region_features = region_features.permute(0, 3, 2, 1).contiguous()
        point_centroid_features = self.pointNet(region_points, True)
        feature_centroid_features = self.pointNet(region_features, False)

        point_centroid_features = point_centroid_features.view(self.batch_size, self.output_channels // 2, -1)
        feature_centroid_features = feature_centroid_features.view(self.batch_size, self.output_channels // 2, -1)
        centroid_features = torch.cat([feature_centroid_features, point_centroid_features], dim=1)

        #unflattening the centroids, because every layer expects points with batch as a separate dimension
        centroids = centroids.view(self.batch_size, 3, -1)
        return centroids, centroid_features

    def pointNet(self, grouped_points, is_raw_points):
        point_features = self.point_point_net(grouped_points) if is_raw_points else self.feature_point_net(grouped_points)
        global_features = torch.max(point_features, dim=2)[0]
        return global_features