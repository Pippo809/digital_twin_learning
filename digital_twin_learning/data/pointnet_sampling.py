import os
import pathlib
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import open3d as o3d
import math
# import telegram_send
import torch
from digital_twin_learning.utils.graph import Edge, Graph, Node, NodeSampling
from digital_twin_learning.utils.helpers import PCDHandler
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt


class PointNetDatasetSampling(Dataset):
    def __init__(
        self,
        pcd_handle: Union[PCDHandler, Path, str],
        cost_path: List[Tuple[str, Dict[str, float]]],
        cache_dir: Union[Path, str] = "",
        use_normals: bool = True,
        mean: np.ndarray = np.zeros(3),
        num_points: int = 1024,
        return_ids: bool = False,
        show_pcd: bool = False,
        metric: str = "success",
        downsample: float = -1.0,
        voxel_size: tuple = (2.0, 2.0, 2.0),
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
        """

        nodes = cost_path

        costs = [NodeSampling.from_tuple(node) for node in nodes]
        position_list = [np.array((node[1]["px"], node[1]["py"], node[1]["pz"])) for node in nodes]
        orientation_list = [np.array((node[1]["qx"], node[1]["qy"], node[1]["qz"], node[1]["qw"])) for node in nodes]
        success_list = [node[1]["success"] for node in nodes]
        self.position_list = np.array(position_list)
        self.success_list = np.array(success_list)
        self.orientation_list = np.array(orientation_list)

        pcd = o3d.io.read_point_cloud(pcd_handle)
        pcd_handle = PCDHandler(pcd, mean, {}, position_list, success_list, orientation_list, voxel_size, overlap)
        self.pcd_handle = pcd_handle

        if downsample > 0:
            self.pcd_handle.subsample_pcd(downsample)
            cache_dir = f"{cache_dir}_downsampled_{str(downsample)}"
        self.show_pcd = show_pcd
        self.costs = costs
        self.use_cache = cache_dir is not None
        self.cache_dir = cache_dir
        self.metric = metric
        if metric != "success":
            max_CoT = np.nanmax([cost.cost for cost in self.costs])
            print("Maximum cost of transport:", max_CoT)
        else:
            self.cache_dir = cache_dir if cache_dir else "None"
        self.use_normals = use_normals
        self.num_points = num_points
        self.min_points = min_points
        self.return_ids = return_ids

        self.verts_list = []
        self.starts_list = []
        self.pose_list = []
        self.center_list = []
        self.cost_list = []
        self.pos_number = []
        self.class_labels = []

        self.preprare_dataset_from_voxels()

    def preprare_dataset_from_voxels(self, keep_invalid=True, save=True):
        """Method to preparate the dataset for the sampling
        Parameters
        ----------
        keep_invalid : [bool]
            flag to indicate if to keep also the invalid voxels (no GT samples) or not
        save : [bool]
            flag to indicate if to save the generated dataset
        """
        stem = "pos_orient_only_valid" if not keep_invalid else "classification_data_with_neg_data_rand"

        if os.path.exists(f"{self.cache_dir}_voxel_sampling_{stem}.pt"):
            print(f"  --> Loading dataset from cache: {self.cache_dir}_voxel_sampling.pt")
            (
                self.verts_list,
                self.starts_list,
                self.pose_list,
                self.cost_list,
                self.center_list,
                self.pos_number,
                self.max_pos_number,
                self.class_labels,
            ) = torch.load(f"{self.cache_dir}_voxel_sampling_{stem}.pt")
            return

        else:
            (num_voxels_x, num_voxels_y, num_voxels_z) = self.pcd_handle.calculate_samples()
            for i in tqdm(range(num_voxels_x)):
                for j in tqdm(range(num_voxels_y), leave=False):
                    for k in range(num_voxels_z):
                        spcd, centers, start_poses, start_orientations, successes, class_label, shifts = self.pcd_handle.sample_from_voxel(i, j, k, False)
                        if len(spcd.points) < self.min_points or len(start_poses) == 0:
                            pass
                        else:
                            ratio = min(1.0, (self.num_points + 20) / len(spcd.points))
                            spcd = spcd.random_down_sample(ratio)
                            replace = False
                            if len(spcd.points) < self.num_points:
                                replace = True

                            vids = torch.tensor(np.random.choice(len(spcd.points), self.num_points, replace=replace))
                            vertices = torch.tensor(np.ascontiguousarray(spcd.points)).float()[vids]
                            spcd = spcd.normalize_normals()

                            if keep_invalid and (len(start_poses) == 0 and np.count_nonzero(successes == 1.0) == 0):
                                self.verts_list.append(vertices)
                                self.starts_list.append(np.array((2, 2, 2)))
                                self.pose_list.append(np.array((0, 0, 0, 1)))
                                self.center_list.append(center)
                                self.cost_list.append(0)
                                self.pos_number.append(0)
                                self.class_labels.append(0)
                            elif len(start_poses) == 0:
                                pass
                            else:
                                for start_pose, start_orientation, success, center, shift in zip(start_poses, start_orientations, successes, centers, shifts):
                                    if success >= 0.0:
                                        points = vertices.data.detach().numpy().copy()
                                        points[:, :3] = points[:, :3] + shift
                                        vertices_shifted = torch.Tensor(points)
                                        self.verts_list.append(vertices_shifted)
                                        self.starts_list.append(start_pose - center + shift)
                                        self.pose_list.append(start_orientation)
                                        self.center_list.append(center)
                                        self.cost_list.append(success)
                                        self.pos_number.append(np.count_nonzero(successes == 1.0))
                                        self.class_labels.append(class_label)

            self.max_pos_number = max(self.pos_number)
            print(len(self.class_labels), np.count_nonzero(self.class_labels), len(self.class_labels) - np.count_nonzero(self.class_labels))
            if save:
                torch.save((self.verts_list,
                            self.starts_list,
                            self.pose_list,
                            self.cost_list,
                            self.center_list,
                            self.pos_number,
                            self.max_pos_number,
                            self.class_labels,
                            ), f"{self.cache_dir}_voxel_sampling_{stem}.pt")

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, index):
        verts = self.verts_list[index]
        if self.return_ids:
            return (
                verts,
                self.starts_list[index],
                self.pose_list[index],
                self.cost_list[index],
                self.pos_number[index],
                self.class_labels[index],
                index,
            )
        else:
            return (
                verts,
                self.starts_list[index],
                self.pose_list[index],
                self.cost_list[index],
                self.pos_number[index],
                self.class_labels[index],
            )

    def draw_histogram(self, numbers):
        # Create a dictionary to store the frequency of each number
        frequency = {}
        for num in numbers:
            if num in frequency:
                frequency[num] += 1
            else:
                frequency[num] = 1

        # Use matplotlib to draw the histogram
        plt.bar(frequency.keys(), frequency.values())  # type: ignore
        plt.savefig(f"{self.cache_dir}_voxel_sampling_only_pos_frequency.png")

class PCDComplete(Dataset):
    def __init__(
            self,
            pcd_handle: Union[PCDHandler, Path, str],
            voxel_size=(2.0, 2.0, 2.0),
            overlap=0.5,
            num_points=1024,
            min_points=50,
    ):
        if type(pcd_handle) == pathlib.PosixPath or type(pcd_handle) == str:
            pcd = o3d.io.read_point_cloud(pcd_handle)
            pcd_handle = PCDHandler(pcd, voxel_size=voxel_size, overlap=overlap)
            assert type(pcd_handle) == PCDHandler, "Error in loading the PCD"
        self.pcd_handle = pcd_handle
        self.verts_list = []
        self.center_list = []
        self.spigoli = []

        # Create a dictionary to store the adjacent voxels for each voxel
        self.adjacent_voxels = {}

        self.idx2voxel = {}
        self.voxel2idx = {}

        self.prepare_dataset(voxel_size, overlap, num_points, min_points)
        print(len(self.verts_list), len(self.verts_list[0]))

    def prepare_dataset(self, voxel_size, overlap, num_points=1024, min_points=100):
        (num_voxels_x, num_voxels_y, num_voxels_z) = self.pcd_handle.calculate_samples()
        n = 0
        for i in tqdm(range(num_voxels_x)):
            for j in tqdm(range(num_voxels_y), leave=False):
                for k in range(num_voxels_z):
                    spcd, center, origin = self.pcd_handle.sample_no_cost(i, j, k, voxel_size, overlap)

                    # Store the adjacent voxels for this voxel in the dictionary
                    self.adjacent_voxels[f'{i}_{j}_{k}'] = []
                    if i > 0:
                        self.adjacent_voxels[f'{i}_{j}_{k}'].append(f'{i-1}_{j}_{k}')
                    if i < num_voxels_x - 1:
                        self.adjacent_voxels[f'{i}_{j}_{k}'].append(f'{i+1}_{j}_{k}')
                    if j > 0:
                        self.adjacent_voxels[f'{i}_{j}_{k}'].append(f'{i}_{j-1}_{k}')
                    if j < num_voxels_y - 1:
                        self.adjacent_voxels[f'{i}_{j}_{k}'].append(f'{i}_{j+1}_{k}')
                    if k > 0:
                        self.adjacent_voxels[f'{i}_{j}_{k}'].append(f'{i}_{j}_{k-1}')
                    if k < num_voxels_z - 1:
                        self.adjacent_voxels[f'{i}_{j}_{k}'].append(f'{i}_{j}_{k+1}')

                    if len(spcd.points) < min_points:
                        pass
                    else:
                        ratio = min(1.0, (num_points + 20) / len(spcd.points))
                        spcd = spcd.random_down_sample(ratio)
                        replace = False
                        if len(spcd.points) < num_points:
                            replace = True

                        vids = torch.tensor(np.random.choice(len(spcd.points), num_points, replace=replace))
                        vertices = torch.tensor(np.ascontiguousarray(spcd.points)).float()[vids]

                        self.verts_list.append(vertices)
                        self.center_list.append(center)
                        self.idx2voxel[n] = f'{i}_{j}_{k}'
                        self.voxel2idx[f'{i}_{j}_{k}'] = n
                        n += 1

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, index):
        verts = self.verts_list[index]
        center = self.center_list[index]
        # adjacent_voxels = self.adjacent_voxels[self.idx2voxel[index]]
        return (verts, center, index)


class PCDCompletePoses(Dataset):
    def __init__(
            self,
            pcd_handle: Union[PCDHandler, Path, str],
            cached_dataset: PCDComplete,
            start_list: torch.Tensor,
            scores: torch.Tensor,
            cache_dir: Union[Path, str],
            save=True,
            num_points=1024
    ):
        if type(pcd_handle) == pathlib.PosixPath or type(pcd_handle) == str:
            pcd = o3d.io.read_point_cloud(pcd_handle)
            pcd_handle = PCDHandler(pcd)
            assert type(pcd_handle) == PCDHandler, "Error in loading the PCD"
        self.pcd_handle = pcd_handle

        self.adjacent_voxels = cached_dataset.adjacent_voxels
        self.idx2voxel = cached_dataset.idx2voxel
        self.voxel2idx = cached_dataset.voxel2idx
        self.center_list = cached_dataset.center_list
        self.spigoli = cached_dataset.spigoli

        self.verts_list = []
        self.start_list = []
        self.goal_list = []
        self.labels_list = []

        self.center_starts = []
        self.center_goals = []
        self.start_score = []
        self.goal_score = []

        self.save = save
        self.num_points = num_points
        self.cache_dir = cache_dir

        self.prepare_dataset(cached_dataset, start_list.cpu(), scores.cpu())

    def prepare_dataset(self, cached_dataset: PCDComplete, start_list, scores):
        for idx in tqdm(range(len(cached_dataset))):
            (verts, center, index) = cached_dataset[idx]

            voxel_name = self.idx2voxel[index]
            for adj_voxel in self.adjacent_voxels[voxel_name]:
                if adj_voxel in self.voxel2idx:
                    if scores[index].item() > 0.20:
                        adj_voxel_idx = self.voxel2idx[adj_voxel]

                        start = start_list[index].numpy()[:3] + self.center_list[index]
                        goal = start_list[adj_voxel_idx].numpy()[:3] + self.center_list[adj_voxel_idx]

                        if idx % 24 == 0:
                            flag = False
                        else:
                            flag = False
                        spcd, start_pose, goal_pose = self.pcd_handle.sample_pcd(start,
                                                                                 goal,
                                                                                 flag,
                                                                                 True,
                                                                                 )

                        if len(spcd.points) < 1:
                            pass

                        else:
                            ratio = min(1.0, (self.num_points + 20) / len(spcd.points))
                            spcd = spcd.random_down_sample(ratio)
                            replace = False
                            if len(spcd.points) < self.num_points:
                                replace = True

                            vids = torch.tensor(np.random.choice(len(spcd.points), self.num_points, replace=replace))
                            vertices = torch.tensor(np.ascontiguousarray(spcd.points)).float()[vids]
                            spcd = spcd.normalize_normals()

                            self.verts_list.append(vertices)
                            self.start_list.append(start_pose)
                            self.goal_list.append(goal_pose)
                            self.center_starts.append(start)
                            self.center_goals.append(goal)
                            self.start_score.append(scores[index].item())
                            self.goal_score.append(scores[adj_voxel_idx].item())
                            self.labels_list.append(0.0)
                    else:
                        pass
                else:
                    pass

        if self.save:
            torch.save(
                (
                    self.verts_list,
                    self.labels_list,
                    self.start_list,
                    self.goal_list,
                    self.center_starts,
                    self.center_goals,
                    self.start_score,
                    self.goal_score,
                ),
                f"{self.cache_dir}_generated.pt",
            )

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, index):
        verts = self.verts_list[index]
        return (verts,
                self.labels_list[index],
                self.start_list[index],
                self.goal_list[index],
                index,
                )
    
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
        """

        if type(pcd_handle) == pathlib.PosixPath or type(pcd_handle) == str:
            pcd = o3d.io.read_point_cloud(pcd_handle)
            pcd_handle = PCDHandler(pcd, mean)
        assert type(pcd_handle) == PCDHandler, "Error in loading the PCD"

        nodes, edges = cost_path
        self.nodedict = nodes.pose_dict

        costs = [Edge(int(edge[0]), int(edge[1]), **edge[2], metric=metric) for edge in edges]  # transform the cost in an edge object

        self.pcd_handle = pcd_handle
        if downsample > 0:
            self.pcd_handle.subsample_pcd(downsample)  # type: ignore
            cache_dir = f"{cache_dir}_downsampled_{str(downsample)}"
        self.show_pcd = show_pcd
        self.costs = costs
        self.use_cache = cache_dir is not None
        self.cache_dir = cache_dir
        self.metric = metric
        if metric != "success":
            max_CoT = np.nanmax([cost.cost for cost in self.costs])
            print("Maximum cost of transport:", max_CoT)
        else:
            self.cache_dir = cache_dir if cache_dir else "None"
        self.use_normals = use_normals
        self.num_points = num_points
        self.return_ids = return_ids

        self.verts_list = []
        self.labels_list = []
        self.starts_list = []
        self.goals_list = []
        self.normal_list = []

        self.center_starts = []
        self.center_goals = []
        self.spigoli_start = []
        self.spigoli_goal = []

        if self.costs is not None:
            self.preprare_dataset_from_costs(save_txt=False)

    def preprare_dataset_from_costs(self, save=True, save_txt=False):
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