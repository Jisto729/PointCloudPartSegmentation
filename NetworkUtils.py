import torch
from torch_geometric.nn import fps, radius
from torch_geometric.utils import to_dense_batch

def sample(points: torch.tensor, batch_tensor: torch.tensor, batch_size):
    """
    Using furthest point sampling, returns a tensor with indices of quarter of the input points
    Args:
        points: Point to be sampled of shape [batch_size*point_num, 3]
        batch_tensor: batch tensor for points
        batch_size: batch size

    Returns: Tensor of shape [btch_size*point_num*centroid_ratio]

    """
    centroid_ratio = 0.25
    return fps(points, batch=batch_tensor, ratio=centroid_ratio, random_start=True, batch_size= batch_size)


def group(ball_radius, centroids: torch.tensor, points: torch.tensor, point_features: torch.tensor,
          batch_tensor: torch.tensor, centroid_batch_tensor: torch.tensor, max_group_size: int, batch_size: int):
    """
    Groups points and their features around centroids according to the ball radius
    Args:
        ball_radius: ball query radius
        centroids: centroids of the groups of shape [batch_size*centroid_num, 3]
        points: points to be grouped around centroids of shape [batch_size*point_num, 3]
        point_features: features of the grouped points of shape [batch_size*point_num, feature_dim]
        batch_tensor: batch tensor for points
        centroid_batch_tensor: batch tensor for centroids
        max_group_size: maximum number of points to be grouped around centroids
        batch_size: batch size

    Returns: Grouped points of shape [batch_size, centroid_num, max_centroid_group_size, feature_dim]
    """

    #ball query around each centroid within the ball radius
    #result[0] contains a mask of the points belonging to each centroid
    #result[1] contains the indices of the points belonging to each centroid
    result = radius(points, centroids, ball_radius, batch_x=batch_tensor, batch_y=centroid_batch_tensor ,
                    max_num_neighbors=max_group_size)

    #concatenating a point's position to its features for more accurate learning
    concatenated_features = torch.cat([point_features, points], dim=1)

    #created a tensor of the grouped points
    grouped_points, _ = to_dense_batch(concatenated_features[result[1]], result[0], max_num_nodes=max_group_size)

    #reshaping the tensors back
    feature_dimensions = point_features.shape[1]
    grouped_points = grouped_points.view(batch_size, -1, max_group_size, feature_dimensions + 3)
    centroids = centroids.view(batch_size, -1, 3)

    #computing positions for each point around a centroid compared to its group centroid
    grouped_points[:, :, :, -3:] = (grouped_points[:, :, :, -3:] - centroids.unsqueeze(2)) / ball_radius

    return grouped_points

def groupMRG(ball_radius, centroids: torch.tensor, points: torch.tensor, point_features: torch.tensor,
          batch_tensor: torch.tensor, centroid_batch_tensor: torch.tensor, max_group_size: int, batch_size: int):
    """
       Groups points and their features around centroids according to the ball radius
       Args:
           ball_radius: ball query radius
           centroids: centroids of the groups of shape [batch_size*centroid_num, 3]
           points: points to be grouped around centroids of shape [batch_size*point_num, 3]
           point_features: features of the grouped points of shape [batch_size*point_num, feature_dim]
           batch_tensor: batch tensor for points
           centroid_batch_tensor: batch tensor for centroids
                   max_group_size: maximum number of points to be grouped around centroids
        batch_size: batch size

       Returns: Grouped points of shape [batch_size, centroid_num, max_centroid_group_size, feature_dim]
       """

    # ball query around each centroid within the ball radius
    # result[0] contains a mask of the points belonging to each centroid
    # result[1] contains the indices of the points belonging to each centroid
    result = radius(points, centroids, ball_radius, batch_x=batch_tensor, batch_y=centroid_batch_tensor,
                    max_num_neighbors=max_group_size)


    grouped_points, _ = to_dense_batch(points[result[1]], result[0], max_num_nodes=max_group_size)
    grouped_features, _ = to_dense_batch(point_features[result[1]], result[0], max_num_nodes=max_group_size)

    # reshaping the tensors back
    feature_dimensions = point_features.shape[1]
    grouped_points = grouped_points.view(batch_size, -1, max_group_size, 3)
    grouped_features = grouped_features.view(batch_size, -1, max_group_size, feature_dimensions)
    centroids = centroids.view(batch_size, -1, 3)

    # computing positions for each point around a centroid compared to its group centroid
    grouped_points[:, :, :, :] = (grouped_points[:, :, :, :] - centroids.unsqueeze(2)) / ball_radius

    return grouped_points, grouped_features