import torch

#jitters each point of the input point clouds by a random amount within the set limits
def jitter(point_cloud, jitter_amount=0.05, min_jitter=0.01, max_jitter=0.01):
    noise = torch.clamp(torch.randn_like(point_cloud) * jitter_amount, min_jitter, max_jitter)
    return point_cloud + noise

#randomly rotates point cloud around the z axis
def random_rotate_around_z(point_cloud):
    rotation_angle = 2 * torch.pi * torch.randn(1)
    cos = torch.cos(rotation_angle).item()
    sin = torch.sin(rotation_angle).item()

    rot_matrix = torch.tensor([
        [cos, 0, sin],
        [0, 1, 0],
        [-sin, 0, cos]], device=point_cloud.device)
    return torch.matmul(point_cloud, rot_matrix)

#randomly scales a point cloud by an amount within set limits
def random_scale(point_cloud, min_scale=0.8, max_scale=1.25):
    scale = torch.empty(1).to(point_cloud.device).uniform_(min_scale, max_scale).item()
    return point_cloud * scale

#appends the 3d coordinates with a fourth coordinate representing the height of each point from the bottom of the point cloud
def height_append(point_cloud):
    bottom = point_cloud[:, 2].min()
    height = point_cloud[:, 2] - bottom
    return torch.cat([point_cloud, height.unsqueeze(1)], dim=1)
