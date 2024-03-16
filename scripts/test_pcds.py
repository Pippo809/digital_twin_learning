# python
import argparse
import os
import time

import matplotlib
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from torch.utils.tensorboard.writer import SummaryWriter

from digital_twin_learning.data.pointnet import PointNetDataset, RandomPCDChange
from digital_twin_learning.utils.graph import Graph, GraphConfig
from digital_twin_learning.utils.helpers import TaskResolver

PCD = TaskResolver.resolve_pcd_path("ETH_HPH")
GRAPH = TaskResolver.resolve_cost_path("ETH_HPH")
CACHE = "/home/lpiglia/git/digital_twin_navigation/digital_twin_learning/digital_twin_learning/results/cached_data/ETH_HPH_all_with_0_samples.pt"
SAVE = "/home/lpiglia/git/digital_twin_navigation/digital_twin_learning/digital_twin_learning/results/tests/HPH"

config = GraphConfig()
nodes = Graph()

config.graph_file = GRAPH
nodes.initialize(config)

costs = nodes.data_list

test_data = PointNetDataset(PCD, (nodes, costs), cache_dir=CACHE, use_normals=False, mean=np.array([0, 0, 0]), num_points=1024, return_ids=True, preprocess=True)
modifier = RandomPCDChange(max_dropout_ratio=0.95, dropout_height=0.3)
for points, labels, start, end, idx in test_data:
    if idx < 300:
        points = points.unsqueeze(0)
        points = points.data.numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[0, :, 0:3])
        # o3d.visualization.draw_geometries([pcd])
        # os.makedirs(f"{SAVE}/{idx}", exist_ok=True)
        o3d.io.write_point_cloud(f"{SAVE}/{idx}.pcd", pcd)
        pcd.clear()
        # points[:, :, 0:3] = modifier.jit_drop(points[:, :, 0:3])
        # pcd.points = o3d.utility.Vector3dVector(points[0, :, 0:3])

        # # o3d.visualization.draw_geometries([pcd])
        # o3d.io.write_point_cloud(f"{SAVE}/{idx}.pcd", pcd)
