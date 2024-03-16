import os
import pathlib
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import open3d as o3d
import math
# import telegram_send
import torch
from digital_twin_learning.utils.graph import Edge, Graph
from digital_twin_learning.utils.helpers import PCDHandler
from torch.utils.data import Dataset
from tqdm import tqdm


class PointNetDataset(Dataset):
    def __init__(
        self,
        pcd_handle: Union[PCDHandler, Path, str],
        cost_path: Tuple[
            Graph, List[Tuple[Union[str, int], Union[str, int], Dict[str, float]]]
        ],
        cache_dir: Union[Path, str] = "",
        use_normals: bool = True,
        mean: np.ndarray = np.zeros(3),
        num_points: int = 1024,
        return_ids: bool = False,
        show_pcd: bool = False,
        metric: str = "success",
        downsample: float = -1.0,
        voxel_size: Tuple[float, float, float] = (2.0, 2.0, 2.0),
        overlap: float = 0.5,
        min_points: int = 100,
        pose_dict: Dict = {},
    ):
        """Dataset class for PointNet training

        Parameters
        ----------
        pcd_handle : Union[PCDHandler, Path, str]
            Path to pcd file
        cost_path : Tuple[Graph, List[Tuple[Union[str, int], Union[str, int], Dict[str, float]]] ]
            Nodes and associated success cost
        cache_dir : Union[Path, str], optional
            Directory to the chaced dataset, by default ""
        use_normals : bool, optional
            use normals in calculating the dataset, by default True
        mean : np.array, optional
            Pcd center, by default np.zeros(3)
        num_points : int, optional
            number of points to sample, by default 1024
        return_ids : bool, optional
            wether to return the ids of the selected points, by default False
        show_pcd : bool, optional
            (only for debugging) show PCD while sampling points, by default False
        metric : str, optional
            What target to use for the loss (Edge success, CoT, etc.)
        downsample : float, optional
            Voxel dimention to downsample the PCD into
        voxel_sizes : tuple, optional
            How big is the voxel to sample around
        """
        voxel_sizes_dict = dict(x_range=voxel_size[0],
                                y_range=voxel_size[1],
                                z_range=voxel_size[2],
                                )

        if type(pcd_handle) == pathlib.PosixPath or type(pcd_handle) == str:
            pcd = o3d.io.read_point_cloud(pcd_handle)
            pcd_handle = PCDHandler(pcd, mean, conf=voxel_sizes_dict)
        assert type(pcd_handle) == PCDHandler, "Error in loading the PCD"

        edges = cost_path
        self.nodedict = pose_dict

        costs = [Edge(int(edge[0]), int(edge[1]), **edge[2], metric=metric) for edge in edges]  # transform the cost in an edge object

        self.pcd_handle = pcd_handle
        if downsample > 0:
            self.pcd_handle.subsample_pcd(downsample)  # type: ignore
            cache_dir = f"{cache_dir}_downsampled_{str(downsample)}"
        self.show_pcd = False
        self.costs = costs
        self.use_cache = cache_dir is not None
        self.cache_dir = cache_dir
        if voxel_size[0] != 2.0:
            self.metric = f"{metric}_{voxel_size[0]}"
        else: 
            self.metric = f"{metric}"
        if metric != "success":
            max_CoT = np.nanmax([cost.cost for cost in self.costs])
            print("Maximum cost of transport:", max_CoT)
        else:
            self.cache_dir = cache_dir if cache_dir else "None"
        if "learned" in self.cache_dir:
            self.sampled = True
        else:
            self.sampled = False
        self.use_normals = use_normals
        self.num_points = num_points
        self.return_ids = return_ids
        self.voxel_sizes = voxel_size

        self.verts_list = []
        self.labels_list = []
        self.starts_list = []
        self.goals_list = []
        self.normal_list = []

        self.center_starts = []
        self.center_goals = []

        if self.costs is not None and not self.sampled:
            self.preprare_dataset_from_costs(save_txt=False)
        elif self.sampled:
            self.preprare_dataset_from_sampling()

    def preprare_dataset_from_costs(self, save=True, save_txt=False):
        """Preprocess the dataset for faster loading

        Parameters
        ----------
        save : bool, optional
            Save the dataset after elaborating it, by default True
        """
        if os.path.exists(f"{self.cache_dir}_{self.metric}.pt"):
            print(f"  --> Loading dataset from cache: {self.cache_dir}_{self.metric}.pt")
            try:
                (
                    self.verts_list,
                    self.labels_list,
                    self.starts_list,
                    self.goals_list,
                ) = torch.load(f"{self.cache_dir}_{self.metric}.pt")
            except ValueError:
                self.verts_list, self.labels_list = torch.load(self.cache_dir)
                self.starts_list = self.goals_list = [torch.zeros(3)] * len(self.verts_list)
                print("Error in chaching the start and goal points")
            return

        else:
            for cost in tqdm(self.costs):
                label = cost.cost
                if math.isnan(label):
                    pass
                else:
                    spcd, start_pose, goal_pose = self.pcd_handle.sample_pcd(  # type: ignore (already handled by assertion)
                        self.nodedict[str(cost.start_id)][:3],
                        self.nodedict[str(cost.goal_id)][:3],
                        self.show_pcd,
                        True,
                    )

                    if len(spcd.points) < 1:
                        pass
                        # vertices = torch.tensor([[0, 0, 1.0, 0, 0, 1.0]] * self.num_points).float()
                        # labels = torch.tensor([label]).float()
                        # spcd = spcd.normalize_normals()
                        # normals = torch.tensor(np.ascontiguousarray(spcd.normals)).float()

                    else:
                        ratio = min(1.0, (self.num_points + 20) / len(spcd.points))
                        spcd = spcd.random_down_sample(ratio)
                        replace = False
                        if len(spcd.points) < self.num_points:
                            replace = True

                        vids = torch.tensor(
                            np.random.choice(
                                len(spcd.points), self.num_points, replace=replace
                            )
                        )
                        vertices = torch.tensor(np.ascontiguousarray(spcd.points)).float()[vids]
                        labels = torch.tensor(np.array([label])).float()
                        spcd = spcd.normalize_normals()
                        normals = torch.tensor(np.ascontiguousarray(spcd.normals)).float()
                        # if self.use_normals:
                        #     vertices = torch.cat((vertices, normals[vids]), dim=-1)

                        self.verts_list.append(vertices)
                        self.labels_list.append(labels)
                        self.starts_list.append(start_pose)
                        self.goals_list.append(goal_pose)
                        self.normal_list.append(normals)
            if save:
                torch.save(
                    (
                        self.verts_list,
                        self.labels_list,
                        self.starts_list,
                        self.goals_list,
                    ),
                    f"{self.cache_dir}_{self.metric}.pt",
                )
                if save_txt:
                    os.makedirs(self.cache_dir, exist_ok=True)
                    for idx, (verts, normals, labels, starts, goals) in enumerate(zip(self.verts_list, self.normal_list, self.labels_list, self.starts_list, self.goals_list)):
                        with open(f"{self.cache_dir}/{idx}.txt", 'w') as f:
                            starts = ",".join([str(coord) for coord in starts.tolist()])
                            goals = ",".join([str(coord) for coord in goals.tolist()])
                            firstline = ",".join([str(labels.tolist()[0]) for i in range(6)])
                            secondline = ",".join([starts, goals])
                            f.write(firstline)
                            f.write("\n")
                            f.write(secondline)
                            f.write("\n")
                            for (vert, normal) in zip(verts, normals):
                                vert = ",".join([str(coord) for coord in vert.tolist()])
                                normal = ",".join([str(coord) for coord in normal.tolist()])
                                line = ",".join([vert, normal])
                                f.write(line)
                                f.write("\n")

    def preprare_dataset_from_sampling(self):
        """Preprocess the dataset for faster loading

        Parameters
        ----------
        save : bool, optional
            Save the dataset after elaborating it, by default True
        """
        if os.path.exists(f"{self.cache_dir}_generated.pt"):
            print(f"  --> Loading dataset from cache: {self.cache_dir}_generated.pt")
            try:
                (
                    self.verts_list,
                    self.labels_list,
                    self.start_list,
                    self.goal_list,
                    self.center_starts,
                    self.center_goals,
                    self.start_score,
                    self.goal_score,
                ) = torch.load(f"{self.cache_dir}_generated.pt")
            except ValueError:
                self.verts_list, self.labels_list = torch.load(self.cache_dir)
                self.starts_list = self.goals_list = [torch.zeros(3)] * len(self.verts_list)
                print("Error in chaching the start and goal points")
            print(len(self.center_starts), len(self.center_goals), len(self.start_list))
            # print(self.start_list[10])
        else:
            raise ValueError("Cached Dataset not found!")

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, index):
        verts = self.verts_list[index]
        if self.sampled:
            if self.return_ids:
                return (
                    verts,
                    self.labels_list[index],
                    self.start_list[index],
                    self.goal_list[index],
                    self.center_starts[index],
                    self.center_goals[index],
                    self.start_score[index],
                    self.goal_score[index],
                    index,
                )
            else:
                return (
                    verts,
                    self.labels_list[index],
                    self.starts_list[index],
                    self.goals_list[index],
                    self.center_starts[index],
                    self.center_goals[index],
                    self.start_score[index],
                    self.goal_score[index],
                )
        elif self.return_ids:
            return (
                verts,
                self.labels_list[index],
                self.starts_list[index],
                self.goals_list[index],
                index,
            )
        else:
            return (
                verts,
                self.labels_list[index],
                self.starts_list[index],
                self.goals_list[index],
            )


class DataModifier:
    """taken from github: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/provider.py
    various data modification for the pointcloud data"""

    @staticmethod
    def normalize_data(batch_data):
        """Normalize the batch data, use coordinates of the block centered at origin,
        Input:
            BxNxC array
        Output:
            BxNxC array
        """
        B, N, C = batch_data.shape
        normal_data = np.zeros((B, N, C))
        for b in range(B):
            pc = batch_data[b]
            centroid = np.mean(pc, axis=0)
            pc = pc - centroid
            m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
            pc = pc / m
            normal_data[b] = pc
        return normal_data

    @staticmethod
    def shuffle_data(data, labels):
        """Shuffle data and labels.
        Input:
        data: B,N,... numpy array
        label: B,... numpy array
        Return:
        shuffled data, label and shuffle indices
        """
        idx = np.arange(len(labels))
        np.random.shuffle(idx)
        return data[idx, ...], labels[idx], idx

    @staticmethod
    def shuffle_points(batch_data):
        """Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
        """
        idx = np.arange(batch_data.shape[1])
        np.random.shuffle(idx)
        return batch_data[:, idx, :]

    @staticmethod
    def rotate_point_cloud(batch_data):
        """Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
        BxNx3 array, original batch of point clouds
        Return:
        BxNx3 array, rotated batch of point clouds
        """
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array(
                [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
            )
            shape_pc = batch_data[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        return rotated_data

    @staticmethod
    def rotate_point_cloud_z(batch_data):
        """Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
        BxNx3 array, original batch of point clouds
        Return:
        BxNx3 array, rotated batch of point clouds
        """
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array(
                [[cosval, sinval, 0], [-sinval, cosval, 0], [0, 0, 1]]
            )
            shape_pc = batch_data[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        return rotated_data

    @staticmethod
    def rotate_point_cloud_with_normal(batch_xyz_normal):
        """Randomly rotate XYZ, normal point cloud.
        Input:
            batch_xyz_normal: B,N,6, first three channels are XYZ, last 3 all normal
        Output:
            B,N,6, rotated XYZ, normal point cloud
        """
        for k in range(batch_xyz_normal.shape[0]):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array(
                [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
            )
            shape_pc = batch_xyz_normal[k, :, 0:3]
            shape_normal = batch_xyz_normal[k, :, 3:6]
            batch_xyz_normal[k, :, 0:3] = np.dot(
                shape_pc.reshape((-1, 3)), rotation_matrix
            )
            batch_xyz_normal[k, :, 3:6] = np.dot(
                shape_normal.reshape((-1, 3)), rotation_matrix
            )
        return batch_xyz_normal

    @staticmethod
    def rotate_perturbation_point_cloud_with_normal(
        batch_data, angle_sigma=0.06, angle_clip=0.18
    ):
        """Randomly perturb the point clouds by small rotations
        Input:
        BxNx6 array, original batch of point clouds and point normals
        Return:
        BxNx3 array, rotated batch of point clouds
        """
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
            Rx = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(angles[0]), -np.sin(angles[0])],
                    [0, np.sin(angles[0]), np.cos(angles[0])],
                ]
            )
            Ry = np.array(
                [
                    [np.cos(angles[1]), 0, np.sin(angles[1])],
                    [0, 1, 0],
                    [-np.sin(angles[1]), 0, np.cos(angles[1])],
                ]
            )
            Rz = np.array(
                [
                    [np.cos(angles[2]), -np.sin(angles[2]), 0],
                    [np.sin(angles[2]), np.cos(angles[2]), 0],
                    [0, 0, 1],
                ]
            )
            R = np.dot(Rz, np.dot(Ry, Rx))
            shape_pc = batch_data[k, :, 0:3]
            shape_normal = batch_data[k, :, 3:6]
            rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
            rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), R)
        return rotated_data

    @staticmethod
    def rotate_point_cloud_by_angle(batch_data, rotation_angle):
        """Rotate the point cloud along up direction with certain angle.
        Input:
        BxNx3 array, original batch of point clouds
        Return:
        BxNx3 array, rotated batch of point clouds
        """
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            # rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array(
                [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
            )
            shape_pc = batch_data[k, :, 0:3]
            rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        return rotated_data

    @staticmethod
    def rotate_point_cloud_by_angle_with_normal(batch_data, rotation_angle):
        """Rotate the point cloud along up direction with certain angle.
        Input:
        BxNx6 array, original batch of point clouds with normal
        scalar, angle of rotation
        Return:
        BxNx6 array, rotated batch of point clouds iwth normal
        """
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            # rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array(
                [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
            )
            shape_pc = batch_data[k, :, 0:3]
            shape_normal = batch_data[k, :, 3:6]
            rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
            rotated_data[k, :, 3:6] = np.dot(
                shape_normal.reshape((-1, 3)), rotation_matrix
            )
        return rotated_data

    @staticmethod
    def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
        """Randomly perturb the point clouds by small rotations
        Input:
        BxNx3 array, original batch of point clouds
        Return:
        BxNx3 array, rotated batch of point clouds
        """
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
            Rx = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(angles[0]), -np.sin(angles[0])],
                    [0, np.sin(angles[0]), np.cos(angles[0])],
                ]
            )
            Ry = np.array(
                [
                    [np.cos(angles[1]), 0, np.sin(angles[1])],
                    [0, 1, 0],
                    [-np.sin(angles[1]), 0, np.cos(angles[1])],
                ]
            )
            Rz = np.array(
                [
                    [np.cos(angles[2]), -np.sin(angles[2]), 0],
                    [np.sin(angles[2]), np.cos(angles[2]), 0],
                    [0, 0, 1],
                ]
            )
            R = np.dot(Rz, np.dot(Ry, Rx))
            shape_pc = batch_data[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
        return rotated_data

    @staticmethod
    def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
        """Randomly jitter points. jittering is per point.
        Input:
        BxNx3 array, original batch of point clouds
        Return:
        BxNx3 array, jittered batch of point clouds
        """
        B, N, C = batch_data.shape
        assert clip > 0
        jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
        jittered_data = np.add(jittered_data, batch_data)
        return jittered_data

    @staticmethod
    def shift_point_cloud(batch_data, shift_range=0.1):
        """Randomly shift point cloud. Shift is per point cloud.
        Input:
        BxNx3 array, original batch of point clouds
        Return:
        BxNx3 array, shifted batch of point clouds
        """
        B, N, C = batch_data.shape
        shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
        for batch_index in range(B):
            batch_data[batch_index, :, :] += shifts[batch_index, :]
        return batch_data

    @staticmethod
    def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
        """Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
        """
        B, N, C = batch_data.shape
        scales = np.random.uniform(scale_low, scale_high, B)
        for batch_index in range(B):
            batch_data[batch_index, :, :] *= scales[batch_index]
        return batch_data

    @staticmethod
    def random_point_dropout(batch_pc, max_dropout_ratio=0.875, dropout_height=0.2):
        """batch_pc: BxNx3"""
        for b in range(batch_pc.shape[0]):
            dropout_ratio = max_dropout_ratio  # np.random.random() * max_dropout_ratio
            drop_idx_0 = (np.random.random((batch_pc.shape[1])) <= dropout_ratio)
            delta = abs(np.amax(batch_pc[b, :, 2]) - np.amin(batch_pc[b, :, 2]))
            drop_idx_1 = ((batch_pc[b, :, 2] - np.amin(batch_pc[b, :, 2])) / delta <= dropout_height)
            drop_idx = np.nonzero(np.logical_and(drop_idx_0, drop_idx_1))
            if dropout_height >= 1.0:
                drop_idx = np.nonzero(drop_idx_0)
            if len(drop_idx) > 0:
                batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
        return batch_pc


class RandomPCDChange(DataModifier):
    def __init__(self, max_s=0.2, max_c=0.3, max_dropout_ratio=0.75, dropout_height=0.2):
        self.max_s = max_s
        self.max_c = max_c
        self.max_dropout_ratio = max_dropout_ratio
        self.dropout_height = dropout_height

    def random_jitter(self, batch_data):
        """Randomly jitter points with different amounts per iteration.
        Input:
        BxNx3 array, original batch of point clouds
        Return:
        BxNx3 array, jittered batch of point clouds
        """
        sigma = self.max_s  # np.random.random() * self.max_s
        clip = self.max_c  # np.random.random() * self.max_c
        return super().jitter_point_cloud(batch_data, sigma=sigma, clip=clip)

    def jit_drop(self, batch_data):
        batch_data = self.random_jitter(batch_data)
        return super().random_point_dropout(batch_data, self.max_dropout_ratio, self.dropout_height)
