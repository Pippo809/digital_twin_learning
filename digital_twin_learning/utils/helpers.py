# pyright: reportGeneralTypeIssues=false
import argparse
import glob
import os
import pathlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import open3d as o3d
# import telegram_send
import torch
from PIL import Image
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from scipy.spatial.transform import Rotation as R

from digital_twin_learning import (DIGITAL_TWIN_ENVS_DIR,
                                   DIGITAL_TWIN_LEARNING_ROOT_DIR,
                                   DIGITAL_TWIN_LOG_DIR)
from digital_twin_learning.models.dgcnn import DGCNN
from digital_twin_learning.models.KPConv.architectures import KPFCNN
from digital_twin_learning.models.KPConv.create_model import KPinit
from digital_twin_learning.models.ldgnn import LDGCNN

"""
Parsing configurations.
"""


def get_args() -> argparse.Namespace:
    """Defines custom command-line arguments and parses them.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    custom_parameters = [
        {
            "name": "--checkpoint_path",
            "type": str,
            "default": f"{DIGITAL_TWIN_LOG_DIR}/trained_models/"
        },
        {
            "name": "--data_cachedir",
            "type": str,
            "default": f"{DIGITAL_TWIN_LOG_DIR}/cached_data/"
        },
        {
            "name": "--mean",
            "type": str,
            "default": "0.,0.,0.",
            "help": "mean to be added to the sampled PCDs"
        },
        {
            "name": "--evaluate_only",
            "action": "store_true",
            "default": False,
            "help": "evaluate using pretrained model",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided.",
        },
        {
            "name": "--use_normals",
            "action": "store_true",
            "default": False,
            "help": "to use normals for the classification or not",
        },
        {
            "name": "--world",
            "type": str,
            "default": "nuketown",
            "help": "world for which you want to train/predict the costs",
        },
        {
            "name": "--model_name",
            "type": str,
            "default": "dgcnn",
            "choices": ["dgcnn"],
            "help": "model architecture, add other models here",
        },
        {
            "name": "--sample",
            "action": "store_true",
            "default": False,
            "help": "If you want to run the sampling repo",
        },
        {
            "name": "--save_pred",
            "action": "store_true",
            "default": False,
            "help": "Save the predicted costs or not",
        },
        {
            "name": "--startified",
            "action": "store_true",
            "default": False,
            "help": "Use stratified random sampling for the dataset or not",
        },
        {
            "name": "--cuda_id",
            "type": int,
            "default": 0,
            "help": "Cuda device id",
        },
        {
            "name": "--cached_model",
            "type": str,
            "default": "",
            "help": "cached model name, if the model does not exist the network will create a new one with this name; otherwise it will use the existing one as a checkpoint",
        },
        {
            "name": "--cached_dataset",
            "type": str,
            "default": "",
            "help": "cached dataset name"
        },
        {
            "name": "--num_samples",
            "type": int,
            "default": 0,
            "help": "Number of samples, 0 = all samples",
        },
        {
            "name": "--num_epochs",
            "type": int,
            "default" : 3,
            "help": "Number of epochs",
        },
        {
            "name": "--use_poses",
            "default" : False,
            "help": "Feed the poses to the training pipeline",
            "action": "store_true"
        },
        {
            "name": "--tf_metric",
            "type": float,
            "help": "Threshold for a sample to be considered True",
            "default" : 0.6,
        },
        {
            "name": "--epoch_model",
            "type": str,
            "help": "Epoch from which to resume the training, empty --> final",
            "default" : "",
        },
        {
            "name": "--metric",
            "type": str,
            "help": "metric for the label to use for prediction",
            "default" : "success",
        },
        {
            "name": "--finetune",
            "action": "store_true",
            "help": "freeze the first layers and train only the MLP",
            "default" : False,
        },
        {
            "name": "--downsample",
            "type": float,
            "help": "Quantity to voxel downsample the PointCloud",
            "default" : -1.0,
        },
        {
            "name": "--jitter_train",
            "action": "store_true",
            "help": "Randomly jitter the points in the training",
            "default" : False,
        },
        {
            "name": "--jitter_test",
            "action": "store_true",
            "help": "Randomly jitter the points in the testing",
            "default" : False,
        },
        {
            "name": "--max_dropout_ratio",
            "type": float,
            "help": "Max ratio of points to be dropped out (only applied if jitter is active)",
            "default" : 0.5,
        },
        {
            "name": "--dropout_height",
            "type": float,
            "help": "Maximum height for the points to be dropped out (only applied if jitter is active)",
            "default" : 0.3,
        },
        {
            "name": "--max_s",
            "type": float,
            "help": "Scale of the jittering (only applied if jitter is active)",
            "default" : 0.2,
        },
        {
            "name": "--max_c",
            "type": float,
            "help": "Maximimum delta for the jittered points, clips the maximum jitter to such value (only applied if jitter is active)",
            "default" : 0.3,
        },
        {
            "name": "--voxel_size_x",
            "type": float,
            "help": "Size of the sampled voxels",
            "default" : 2.0,
        },
        {
            "name": "--voxel_size_y",
            "type": float,
            "help": "Size of the sampled voxels",
            "default" : 2.0,
        },
        {
            "name": "--voxel_size_z",
            "type": float,
            "help": "Size of the sampled voxels",
            "default" : 2.0,
        },
        {
            "name": "--overlap",
            "type": float,
            "help": "Overlap between the sampled voxels",
            "default" : 0.2,
        },
        {
            "name": "--encoder_model",
            "type": str,
            "default": "",
        },
        {
            "name": "--stem",
            "type": str,
            "help": "stem to add to the saved dataset, used for the sampling generation if generating multiple dataset and want to differentiate their names",
            "default": "",
        },
    ]

    # parse arguments
    parser = argparse.ArgumentParser(description="PoinCloud Learning")
    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    if "choices" in argument:
                        parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], choices=argument["choices"], help=help_str)
                    else:
                        parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], help=help_str)
                else:
                    parser.add_argument(argument["name"], type=argument["type"], help=help_str)
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)

        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()
    args = parser.parse_args()

    args.checkpoint_path = f"{DIGITAL_TWIN_LOG_DIR}/trained_models"
    args.data_cachedir = f"{DIGITAL_TWIN_LOG_DIR}/cached_data"
    args.pred_path = f"{DIGITAL_TWIN_LOG_DIR}/predictions"
    args.cached_model_path = f"{DIGITAL_TWIN_LOG_DIR}/trained_models/{args.cached_model}"
    if args.encoder_model != "":
        args.cached_encoder_path = f"{DIGITAL_TWIN_LOG_DIR}/trained_models/{args.encoder_model}"
    else: args.cached_encoder_path = ""
    if args.cached_dataset == "":
        args.cached_dataset = args.world
    if args.epoch_model == "":
        args.epoch_model = "final"
    else:
        args.epoch_model = f"epoch_{args.epoch_model}"
    if args.stem != "":
        args.stem = f"_{args.stem}"
    else: args.stem = ""

    args.rand_pcds = {"max_dropout_ratio": args.max_dropout_ratio, "dropout_height": args.dropout_height, "max_s": args.max_s, "max_c": args.max_c}
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.voxel_size = tuple((args.voxel_size_x, args.voxel_size_y, args.voxel_size_z))

    return args


class TaskResolver:
    """Resolvs the environment assignment etc."""
    def __init__(self, args):
        # dictionaries for saving configurations.
        self.model = args.model_name
        self.sample = args.sample
        self.name = args.world

    def resolve_model_class(self):
        if self.model == "ldgcnn":
            return {
                "ldgcnn": LDGCNN,
            }[self.model]
        elif self.model == "dgcnn":
            return {
                "dgcnn": DGCNN,
            }[self.model]

    def resolve_pcd_path(self) -> str:
        paths = glob.glob(f"{DIGITAL_TWIN_ENVS_DIR}/clouds/*")
        pcds = {
            "generated": f"{DIGITAL_TWIN_ENVS_DIR}/clouds/generated.pcd",
            "generated++": f"{DIGITAL_TWIN_ENVS_DIR}/clouds/generated.pcd",
            "ETH_LEE": f"{DIGITAL_TWIN_ENVS_DIR}/clouds/ETH_LEE_H_with_terrace.pcd",
            "ETH_LEE_cropped": f"{DIGITAL_TWIN_ENVS_DIR}/clouds/ETH_LEE_cropped.pcd",
            "nuketown": f"{DIGITAL_TWIN_ENVS_DIR}/clouds/nuketown_18dense.pcd",
        }
        for path in paths:
            if path not in pcds.values():
                name = os.path.basename(path)
                pcds[name[:-4]] = path
        return pcds[self.name]

    def resolve_cost_path(self) -> str:
        costs = {
            "generated": f"{DIGITAL_TWIN_ENVS_DIR}/graphs/2022-09-08_13-52-07_generated_description/final.graphml.xml",
            "generated++": f"{DIGITAL_TWIN_ENVS_DIR}/graphs/2022-10-21_09-22-50_generated_10_iterations-20221021T154241Z-001/2022-10-21_09-22-50_generated_10_iterations/final.graphml.xml",
            "ETH_LEE": f"{DIGITAL_TWIN_ENVS_DIR}/graphs/2022-09-08_16-25-35_ETH_LEE_H_with_terrace_description/final.graphml.xml",
            "ETH_LEE_cropped": f"{DIGITAL_TWIN_ENVS_DIR}/graphs/ETH_LEE_correct_CoT/final.graphml.xml",
            "ARCHE": f"{DIGITAL_TWIN_ENVS_DIR}/graphs/2022-09-08_18-30-44_ARCHE_description/final.graphml.xml",
            "ETH_HPH": f"{DIGITAL_TWIN_ENVS_DIR}/graphs/ETH_HPH_correct_CoT/final.graphml.xml",
            "mesh_0": f"{DIGITAL_TWIN_ENVS_DIR}/graphs/2022-10-25_18-04-29_0_mesh_description/final.graphml.xml",
            "nuketown": f"{DIGITAL_TWIN_ENVS_DIR}/graphs/2022-10-26_16-42-34_Nuketown_cropped_description/final.graphml.xml",
            "mix": f"{DIGITAL_TWIN_ENVS_DIR}/graphs/2022-11-02_12-54-55_mix_description/final.graphml.xml",
            "mix2": f"{DIGITAL_TWIN_ENVS_DIR}/graphs/2022-11-04_11-29-51_mix2_description/final.graphml.xml",
            "mix3": f"{DIGITAL_TWIN_ENVS_DIR}/graphs/mix3_10iter/final.graphml.xml",
            "rand_gen_0": f"{DIGITAL_TWIN_ENVS_DIR}/graphs/2022-11-28_17-47-27_rand_gen_description/final.graphml.xml",
            "rand_gen_2": f"{DIGITAL_TWIN_ENVS_DIR}/graphs/rand_gen_2_correct_CoT/2022-12-14_15-18-20_rand_gen_2_description/final.graphml.xml",
            "rand_gen_easy": f"{DIGITAL_TWIN_ENVS_DIR}/graphs/2022-12-02_14-47-40_rand_gen_easy_description/final.graphml.xml",
            "extremely_easy": f"{DIGITAL_TWIN_ENVS_DIR}/graphs/2022-12-07_11-44-19_extremely_easy_description/final.graphml.xml",
        }
        paths = glob.glob(f"{DIGITAL_TWIN_ENVS_DIR}/graphs/*")
        for path in paths:
            if f"{path}/final.graphml.xml" not in costs.values():
                name = os.path.basename(path)
                costs[name] = f"{path}/final.graphml.xml"
        return costs[self.name]

    def resolve_mesh_path(self) -> str:
        """Takes the name of the world and outputs the corresponding mesh path
        Parameters
        ----------
        name : str,
            Name of the world used for training/eval
        """
        meshes = {
            "generated": f"{DIGITAL_TWIN_ENVS_DIR}/meshes/generated",
            "generated++": f"{DIGITAL_TWIN_ENVS_DIR}/meshes/generated",
            "ETH_LEE": f"{DIGITAL_TWIN_ENVS_DIR}/meshes/ETH_LEE_H_with_terrace",
            "ETH_LEE_cropped": f"{DIGITAL_TWIN_ENVS_DIR}/meshes/ETH_LEE_H_with_terrace_cropped",
            "ARCHE": f"{DIGITAL_TWIN_ENVS_DIR}/meshes/ARCHE",
            "ETH_HPH": f"{DIGITAL_TWIN_ENVS_DIR}/meshes/ETH_HPH",
            "mesh_0": f"{DIGITAL_TWIN_ENVS_DIR}/meshes/0_mesh",
            "mesh_1": f"{DIGITAL_TWIN_ENVS_DIR}/meshes/1_mesh",
            "nuketown": f"{DIGITAL_TWIN_ENVS_DIR}/meshes/Nuketown_small",
        }
        paths = glob.glob(f"{DIGITAL_TWIN_ENVS_DIR}/meshes/*")
        for path in paths:
            if path not in meshes.values():
                name = os.path.basename(path)
                meshes[name] = path
        return meshes[self.name]

    def get_cfg(self) -> Tuple[object, str, str, str]:
        model_class = self.resolve_model_class()
        pcd_path = self.resolve_pcd_path()
        mesh_path = self.resolve_mesh_path()
        cost_path = self.resolve_cost_path()
        return (model_class, pcd_path, mesh_path, cost_path)

    def to_args(self, args):
        args.model, args.pcd_path, args.mesh_dir, args.cost_path = self.get_cfg()
        return args

    def make_paths(self, args):
        if not os.path.exists(args.data_cachedir):
            os.makedirs(str(args.data_cachedir))
        if not os.path.exists(args.pred_path):
            os.makedirs(args.pred_path)


class PCDHandler:

    default_config = dict(
        x_range=1.0,
        y_range=1.0,
        z_range=1.0,
    )

    def __init__(
        self,
        pcd,
        mean: np.ndarray = np.zeros(3, dtype="float64"),
        conf: dict = {},
        position_list=[],
        success_list=[],
        orientation_list=[],
        voxel_size: tuple = (2.0, 2.0, 2.0),
        overlap: float = 0.5,
    ):

        self.mean = mean
        self.conf = {**self.default_config, **conf}
        self.position_list = np.array(position_list)
        self.success_list = np.array(success_list)
        self.orientation_list = np.array(orientation_list)
        self.voxel_size = voxel_size
        self.overlap = overlap
        self._init(pcd)

    def _init(self, pcd):
        if type(pcd) == pathlib.PosixPath or type(pcd) == str:
            self.pcd = o3d.io.read_point_cloud(pcd)
        elif type(pcd) == o3d.geometry.PointCloud:
            self.pcd = pcd
        else:
            raise TypeError(f"{pcd} cannot be loaded as a pcd")

    def subsample_pcd(self, voxel_size=(1 / 30)):
        downpcd = self.pcd.voxel_down_sample(voxel_size=voxel_size)  # type: ignore
        self.pcd = downpcd

    def sample_pcd(self, start: np.ndarray, end: np.ndarray, show=False, shift_and_rotate=False):
        """Sample points around the start and goal pose

        Parameters
        ----------
        start : np.ndarray
            start position
        end : np.ndarray
            goal position
        show : bool, optional
            (for debugging) wether to show the processed PCDs, by default False
        shift_and_rotate : bool, optional
            Align the PCDs to the start/goal poses, by default False

        Returns
        -------
        pcd
            Sampled PCD
        start, goal
            Coordinates relative to the sampled PCD center
        """
        start = start + self.mean
        end = end + self.mean
        bbPoints = self.create_bounding_box_points(start, end)
        bbBox = o3d.geometry.OrientedBoundingBox().create_from_points(
            o3d.utility.Vector3dVector(bbPoints)
        )

        pcdsub = self.pcd.crop(bbBox)

        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        pcdcenter = (start + end) / 2.0

        if shift_and_rotate:
            pcdsub, pcdcenter, R = self.shift_and_rotate_pcd(pcdsub, start, end)

        if show:
            mesh1 = o3d.geometry.TriangleMesh.create_coordinate_frame(
                origin=R @ (start - pcdcenter)
            )
            mesh2 = o3d.geometry.TriangleMesh.create_coordinate_frame(
                origin=R @ (end - pcdcenter)
            )

            mesh_box1 = o3d.geometry.TriangleMesh.create_box(
                width=0.8, height=0.4, depth=0.7
            )
            mesh_box2 = o3d.geometry.TriangleMesh.create_box(
                width=0.8, height=0.4, depth=0.7
            )

            mesh_box1.translate(R @ (start - pcdcenter))
            mesh_box2.translate(R @ (end - pcdcenter))

            bbBox.translate(-pcdcenter)
            # bbBox = o3d.geometry.OrientedBoundingBox().create_from_points(pcdsub.points)
            bbBox.rotate(R, center=(0, 0, 0.0))

            points = [R @ (start - pcdcenter), R @ (end - pcdcenter)]
            lines = [[0, 1]]
            colors = [[0.7, 0, 0] for i in range(len(lines))]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)

            mesh_box1.compute_vertex_normals()
            mesh_box2.compute_vertex_normals()
            mesh_box1.paint_uniform_color([0.7, 0.1, 0.1])
            mesh_box2.paint_uniform_color([0.1, 0.1, 0.7])
            o3d.visualization.draw_geometries(
                [pcdsub, bbBox, mesh1, mesh2, mesh_box1, mesh_box2, line_set]
            )

        start, end = R @ (start + self.mean - pcdcenter), R @ (end + self.mean - pcdcenter)

        return pcdsub, start, end

    def shift_pcd(self, subpcd, pcdcenter):

        # translate
        subpcd = subpcd.translate(-pcdcenter, relative=True)

        return subpcd

    def shift_and_rotate_pcd(self, subpcd, start, end):
        pcdcenter = (start + end) / 2.0

        # translate
        subpcd = subpcd.translate(-pcdcenter, relative=True)

        # rotate
        zrotate = np.arctan2(end[1] - pcdcenter[1], end[0] - pcdcenter[0])
        R = subpcd.get_rotation_matrix_from_xyz((0, 0.0, -zrotate))
        subpcd.rotate(R, center=(0, 0, 0))

        return subpcd, pcdcenter, R

    def create_bounding_box_points(self, start, end):
        x, y, z = self.conf["x_range"], self.conf["y_range"], self.conf["z_range"]
        points = np.array(
            [
                [-x, -y, z],
                [-x, y, z],
                [x, -y, z],
                [-x, y, z],
                [-x, -y, -z],
                [-x, y, -z],
                [x, -y, -z],
                [-x, y, -z],
            ]
        )

        a = start + points
        b = end + points

        frontvec = end - start
        frontvec[2] = 0.0
        frontvec = frontvec / np.linalg.norm(frontvec)
        upvec = np.array([0, 0, 1.0])
        rightvec = np.cross(frontvec, upvec)

        frontvec = self.conf["x_range"] * frontvec
        upvec = self.conf["y_range"] * upvec
        rightvec = self.conf["z_range"] * rightvec

        points = np.array(
            [
                frontvec + rightvec + upvec,
                frontvec - rightvec + upvec,
                -frontvec + rightvec + upvec,
                -frontvec - rightvec + upvec,
                frontvec + rightvec - upvec,
                frontvec - rightvec - upvec,
                -frontvec + rightvec - upvec,
                -frontvec - rightvec - upvec,
            ]
        )
        a = start + points
        b = end + points

        return np.concatenate([a, b], axis=0)

    def show_pcd(self):
        o3d.visualization.draw_geometries([self.pcd])

    def save_pcd(self, path):
        o3d.io.write_point_cloud(path, self.pcd)

    def calculate_samples(self):
        (x, y, z) = self.voxel_size

        points = np.asarray(self.pcd.points)

        # Divide the point cloud into voxels
        min_x = np.min(points[:, 0])
        max_x = np.max(points[:, 0])
        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])
        min_z = np.min(points[:, 2])
        max_z = np.max(points[:, 2])

        num_voxels_x = int((max_x - min_x) // (x - self.overlap))
        num_voxels_y = int((max_y - min_y) // (y - self.overlap))
        num_voxels_z = int((max_z - min_z) // (z - self.overlap))

        self.min_x = min_x
        self.min_y = min_y
        self.min_z = min_z
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z

        return (num_voxels_x, num_voxels_y, num_voxels_z)

    def sample_from_voxel(self, i, j, k, plot=False, scale=1.0):
        voxel_size = self.voxel_size
        overlap = self.overlap

        points = np.asarray(self.pcd.points)

        voxel_points = points[(points[:, 0] >= self.min_x + i * (voxel_size[0] - overlap) - overlap) &
                              (points[:, 0] < self.min_x + (i + 1) * (voxel_size[0] - overlap) + overlap) &
                              (points[:, 1] >= self.min_y + j * (voxel_size[1] - overlap) - overlap) &
                              (points[:, 1] < self.min_y + (j + 1) * (voxel_size[1] - overlap) + overlap) &
                              (points[:, 2] >= self.min_z + k * (voxel_size[2] - overlap) - overlap) &
                              (points[:, 2] < self.min_z + (k + 1) * (voxel_size[2] - overlap) + overlap)]

        voxel_pcd = o3d.geometry.PointCloud()
        voxel_pcd.points = o3d.utility.Vector3dVector(voxel_points)

        center = np.array(((self.min_x + (i) * (voxel_size[0] - overlap) + voxel_size[0] / 2),
                           (self.min_y + (j) * (voxel_size[1] - overlap) + voxel_size[1] / 2),
                           (self.min_z + (k) * (voxel_size[2] - overlap) + voxel_size[2] / 2),
                           ))

        valid_poses, valid_success, valid_orientations = self.filter_points_in_box_np(center, voxel_size)
        pcdsub = self.shift_pcd(voxel_pcd, center)
        valid_poses, centers, shifts = self.rand_shift(valid_poses, center, scale)
        count = np.count_nonzero(valid_success == 1.0)
        class_label = 0
        if count > 0:
            class_label = 1
        if len(valid_poses) == 0:
            return pcdsub, center, [], [], [], 0, []

        if plot and np.random.random() > 0.95:
            boxes = self.create_boxes(valid_poses[valid_success == 1.0], valid_orientations, center, valid_success)
            o3d.visualization.draw_geometries([self.pcd, *boxes])

        return pcdsub, centers, valid_poses, valid_orientations, valid_success, class_label, shifts

    def filter_points_in_box_np(self, center, box_dimensions):
        points = self.position_list
        success_list = self.success_list
        orientation_list = self.orientation_list
        center = np.array(center)
        box_dimensions = np.array(box_dimensions)
        lower_bounds = center - box_dimensions / 2
        upper_bounds = center + box_dimensions / 2
        in_box = np.all((points >= lower_bounds) & (points <= upper_bounds), axis=1)
        return points[in_box], success_list[np.nonzero(in_box)], orientation_list[np.nonzero(in_box)]
    
    def sample_no_cost(self, i, j, k, voxel_size=(2, 2, 4), overlap=0.5):

        # points = self.points
        points = np.asarray(self.pcd.points)

        voxel_points = points[(points[:, 0] >= self.min_x + i * (voxel_size[0] - overlap) - overlap) &
                            (points[:, 0] < self.min_x + (i + 1) * (voxel_size[0] - overlap) + overlap) &
                            (points[:, 1] >= self.min_y + j * (voxel_size[1] - overlap) - overlap) &
                            (points[:, 1] < self.min_y + (j + 1) * (voxel_size[1] - overlap) + overlap) &
                            (points[:, 2] >= self.min_z + k * (voxel_size[2] - 0.0) - 0.0) &
                            (points[:, 2] < self.min_z + (k + 1) * (voxel_size[2] - 0.0) + 0.0)]

        voxel_pcd = o3d.geometry.PointCloud()
        voxel_pcd.points = o3d.utility.Vector3dVector(voxel_points)

        center = np.array(((self.min_x + (i) * (voxel_size[0] - overlap) + voxel_size[0] / 2),
                           (self.min_y + (j) * (voxel_size[1] - overlap) + voxel_size[1] / 2),
                           (self.min_z + (k) * (voxel_size[2] - 0.0) + voxel_size[2] / 2),
                           ))

        origin = np.array((self.min_x + i * (voxel_size[0] - overlap),
                           self.min_y + j * (voxel_size[1] - overlap),
                           self.min_z + k * (voxel_size[2] - 0.0)))

        pcdsub = self.shift_pcd(voxel_pcd, center)

        return pcdsub, center, origin

    @staticmethod
    def rand_shift(valid_poses, center, scale=1.0):
        new_poses = []
        new_centers = []
        shifts = []
        for valid_pose in valid_poses:
            rand_shift = (np.random.rand(3) - 0.5) * scale
            new_pose = valid_pose + rand_shift
            new_center = center + rand_shift
            new_poses.append(new_pose)
            new_centers.append(new_center)
            shifts.append(rand_shift)

        return new_poses, new_centers, shifts

    @staticmethod
    def create_boxes(coordinates, orientations, center, successes):
        boxes = []
        print(center, len(successes))
        for coordinate, orientation, success, _ in zip(coordinates, orientations, successes, range(15)):
            r = R.from_quat(orientation).as_matrix()
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=(coordinate))
            mesh.rotate(r)
            mesh_box = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.1)
            mesh_box.translate((coordinate))
            mesh_box.rotate(r)
            mesh_box.compute_vertex_normals()
            mesh_box.paint_uniform_color([0.7, 0.1, 0.1])
            boxes.append(mesh)
            boxes.append(mesh_box)
            print(coordinate-center, orientation, success)
        return boxes


def mergeDictionary(dict_1, dict_2):
    dict_3 = {**dict_1, **dict_2}
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = np.append(value, dict_1[key], axis=1)
    return dict_3

 
def tensorboard_img(path: str, name: str, tb: SummaryWriter) -> None:
    """Load an image and save it to Tensorflow"""

    convert_tensor = transforms.ToTensor()
    img = Image.open(path)
    img = convert_tensor(img)
    tb.add_image(name, img)
    return


def copy_folder(src, dst):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(dst):
        os.makedirs(dst)

    # Iterate over the files and directories in the source folder
    for root, dirs, files in os.walk(src):
        # Create the destination subfolder if it doesn't exist
        subfolder = root.replace(src, dst, 1)
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

        # Copy each file to the destination subfolder
        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(subfolder, file)
            shutil.copy(src_path, dst_path)
