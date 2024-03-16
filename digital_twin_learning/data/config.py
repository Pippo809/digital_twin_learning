import inspect
import os
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from digital_twin_learning.data import PointNetDataset, PointNetDatasetSampling, PCDComplete, PCDCompletePoses
from digital_twin_learning.models.ldgnn import RMSELoss
from digital_twin_learning.utils import Graph, GraphConfig
from digital_twin_learning.data.pointnet import RandomPCDChange
from digital_twin_learning.models.sampling import CVAE, Decoder, Encoder
from digital_twin_learning.models.dgcnn import DGCNN
from digital_twin_learning.utils.train_CVAE import TrainUtilsCVAE
from digital_twin_learning.utils.train_test import TrainUtils
from itertools import chain


@dataclass
class HyperParam():
    """Class to generate the hyperparameters for the training pipeline
    """

    test_split = 0.3
    batch_size = 8  # 8
    num_points = 1024
    k = 60
    emb_dim = 256  # 1024
    emb_dim = 256  # 1024
    dropout = 0.5
    lr = 1e-3
    lr = 1e-3
    decay_rate = 0.8
    decay_every = 1
    beta = 0.005

    # CVAE Params
    dim_latent = 35
    y_dim = 512  # 2*emb_dim
    x_dim = 7  # Set to 7 if use also poses
    H = 256
    kl_weight = 1
    class_weight = 1
    recon_weight = 1
    poscountweight = 1
    acc_metric = RMSELoss()
    acc_metric2 = torch.nn.L1Loss()
    pos_count_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.1])).to("cuda:0")  # torch.nn.PoissonNLLLoss(log_input=True, full=True)  # The output of the network represents a log rate
    metric: str = "success"
    if metric == "CoT": lossfun = torch.nn.SmoothL1Loss(beta=beta)  # torch.nn.MSELoss()
    else: lossfun = torch.nn.BCEWithLogitsLoss()
    max_s: float = -1.0
    max_c: float = -1.0
    max_dropout_ratio: float = -1.0
    dropout_height: float = -1.0  # all overloaded
    # modifier: RandomPCDChange = RandomPCDChange(max_s=max_s, max_c=max_c, max_dropout_ratio=max_dropout_ratio, dropout_height=dropout_height)


class CostConfig():
    """Class to generate the cost/success samples"""

    def __init__(self, cost_path, metric="success", sample=False, evaluate=False) -> None:
        self.metric = metric
        self.sample = sample
        if evaluate and sample: # No need to generate the graph because we will sample it
            return
        graph_config = GraphConfig(graph_file=cost_path)
        graph = Graph()
        graph.initialize(graph_config, sample=sample)

        self.costs_list = graph.data_list
        self.cost_successes = graph.success_list
        self.graph = graph
        self.dataset_generator = PointNetDataset if not sample else PointNetDatasetSampling


    def generate_samples(self, args):
        if args.num_samples > 0:
           
            if args.startified:
                uniques, counts = np.unique(self.cost_successes, return_counts=True)
                min_counts = np.min(counts)
                selected_idx = np.array([]).astype(np.int32)
                for u in uniques:
                    selected_idx = np.concatenate(
                        [
                            selected_idx,
                            np.random.choice(
                                np.where(self.cost_successes == u)[0],
                                size=min_counts,
                                replace=False,
                            ),
                        ],
                        axis=0,
                    )
            else:
                selected_idx = np.arange(len(self.cost_successes))

            selected_idx = np.random.choice(selected_idx, min(args.num_samples, len(selected_idx)), replace=False)

        else:
            selected_idx = np.arange(len(self.cost_successes))

        print(f"{len(selected_idx)} samples will be used for training and validation!")
        if args.metric == "success":
            unique, count = np.unique(self.cost_successes, return_counts=True)
            data = np.vstack((unique, count)).T
            print(
                tabulate(
                    data,
                    headers=["Successes in the GT data", "Number of occurences"],
                    floatfmt=".2f",
                )
            )
        self.costs_list = [self.costs_list[idx] for idx in selected_idx]

    def train_test_split(self, args, test_split: float, num_points=1024):
        """Script to divide the pointcloud and the edges in samples for
        the network

        Parameters
        ----------
        args : argparse.parseargument
            arguments defining paths etc.
        costs : Tuple[Graph, List]
            ground truth costs for the edges
        test_split : float
            % of the dataset used for validation
        num_points : int, optional
            number of points to sample, by default 1024

        Returns
        -------
        train_dataset, test_dataset
            PointNetDatasets for the training and validation
        """

        all_idx = np.arange(len(self.costs_list))
        np.random.RandomState(2021).shuffle(all_idx)
        num_trainids = int(round((1 - test_split) * len(all_idx)))
        train_idx, test_idx = all_idx[:num_trainids], all_idx[num_trainids:]

        train_costs = [self.costs_list[i] for i in train_idx]
        test_costs = [self.costs_list[i] for i in test_idx]

        # train dataset
        train_dataset = self.dataset_generator(
            args.pcd_path,
            train_costs,
            cache_dir=f"{args.data_cachedir}/{args.cached_dataset}/train_with_{args.num_samples}_samples",
            use_normals=args.use_normals,
            mean=np.fromstring(args.mean, dtype="float64", sep=","),
            num_points=num_points,
            show_pcd=False,
            downsample=args.downsample,
            voxel_size=args.voxel_size,
            overlap=args.overlap,
            min_points=30,
            pose_dict=self.graph.pose_dict

        )

        # test dataset
        test_dataset = self.dataset_generator(
            args.pcd_path,
            test_costs,
            cache_dir=f"{args.data_cachedir}/{args.cached_dataset}/test_with_{args.num_samples}_samples",
            use_normals=args.use_normals,
            num_points=num_points,
            mean=np.fromstring(args.mean, dtype="float64", sep=","),
            return_ids=True,
            show_pcd=False,
            downsample=args.downsample,
            voxel_size=args.voxel_size,
            overlap=args.overlap,
            min_points=30,
            pose_dict=self.graph.pose_dict
        )

        return train_dataset, test_dataset

    def generate_dataset(self, args, hyperparams):
        """Function to sample points and save them in a Dataset

        Parameters
        ----------
        args : argparser
            args to be passed with paths etc.
        hyperparams : DataClass
            Class with the Hyperparameters

        Returns
        -------
        PointNetDataset
            Train and Validation Datasets
        """
        os.makedirs(f"{args.data_cachedir}/{args.cached_dataset}", exist_ok=True)
        if args.evaluate_only:
            if self.sample:
                test_data = PCDComplete(args.pcd_path, args.voxel_size, args.overlap, hyperparams.num_points, min_points=30)
                self.test_data = test_data
                return [0, 0, 0], test_data
            else:
                test_data = self.dataset_generator(
                    args.pcd_path,
                    self.costs_list,
                    cache_dir=f"{args.data_cachedir}/{args.cached_dataset}/all_with_{args.num_samples}_samples",
                    use_normals=args.use_normals,
                    num_points=hyperparams.num_points,
                    mean=np.fromstring(args.mean, dtype="float64", sep=","),
                    return_ids=True,
                    show_pcd=False,
                    downsample=args.downsample,
                    voxel_size=args.voxel_size,
                    overlap=args.overlap,
                    min_points=30,
                    pose_dict=self.graph.pose_dict
                )
                train_data = [0, 0, 0]
                print(f"{len(test_data)} samples (all the dataset) are used for testing")
        else:
            self.generate_samples(args)
            train_data, test_data = self.train_test_split(args, hyperparams.test_split, hyperparams.num_points)
            print(f"{len(train_data)} samples will be used for training, {len(test_data)} samples are used for testing")
        return train_data, test_data


class ModelConfig(HyperParam):
    def __init__(self, args) -> None:
        """Class to generate the training model
        """
        super().__init__(metric=args.metric)
        self.sample = args.sample
        self.learned = True if "learned" in args.cached_dataset else False
        if self.learned:
            self.test_batch_size = 1
        else:
            self.test_batch_size = super().batch_size
        if not self.sample:
            self.model = args.model(
                k=super().k,
                emb_dim=super().emb_dim,
                output_channels=1,
                dropout=super().dropout,
                pose_features=args.use_poses,
                use_normals=args.use_normals,
            )
            self.model.to(args.device)

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=super().lr, betas=(0.9, 0.999), eps=1e-08)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=super().decay_every, gamma=super().decay_rate)

            self.save_log(args)
        else:
            dim_latent = super().dim_latent
            y_dim = super().y_dim
            x_dim = super().x_dim
            H = super().H
            self.kl_weight = super().kl_weight
            self.map_encoder = args.model(k=super().k, emb_dim=super().emb_dim, map_dim=512, output_channels=1, pose_features=False, use_normals=False, dropout=0.5)
            self.encoder = Encoder(x_dim + y_dim, H, dim_latent)
            self.decoder = Decoder(dim_latent + y_dim, H, x_dim)
            self.model = CVAE(self.encoder, self.decoder)
            self.model.to(args.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.999), lr=super().lr)
            self.save_log(args)

    def _get_dict(self, args, train_data, test_data):
        if not self.sample:
            out = {
                "model": self.model,
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
                "lossfun": super().lossfun,
                "accuracyfuns": (super().acc_metric, super().acc_metric2),
                "train_loader": DataLoader(train_data, batch_size=super().batch_size, shuffle=True, drop_last=True),
                "test_loader": DataLoader(test_data, batch_size=self.test_batch_size, shuffle=False, drop_last=False),
                "tb": SummaryWriter(f"{args.pred_path}/{args.experiment_name}"),
                "device": args.device,
                "finetune": args.finetune,
                "metric": args.metric,
                "modifier": RandomPCDChange(**args.rand_pcds),
                "learned": self.learned,
            }
            return out
        else:
            out = {
                "model": self.model,
                "encoder": self.encoder,
                "decoder": self.decoder,
                "map_encoder": self.map_encoder,
                "kl_weight": self.kl_weight,
                "class_weight": super().class_weight,
                "recon_weight": super().recon_weight,
                "poscountweight": super().poscountweight,
                "optimizer": self.optimizer,
                "classlossfun": super().lossfun,
                "poscountlossfun": super().pos_count_loss,
                "accuracyfuns": (super().acc_metric, super().acc_metric2),
                "train_loader": DataLoader(train_data, batch_size=super().batch_size, shuffle=True, drop_last=True),
                "test_loader": DataLoader(test_data, batch_size=super().batch_size, shuffle=False, drop_last=False),
                "tb": SummaryWriter(f"{args.pred_path}/{args.experiment_name}"),
                "device": args.device,
                "finetune": args.finetune,
                "metric": args.metric,
            }
            return out
    
    def _getTrainUtilsClass(self):
        if not self.sample:
            return TrainUtils
        else:
            return TrainUtilsCVAE


    def save_log(self, args):
        """Generates a log with all the settings of the training and saves it in a .txt

        Args:
            args (Argparser): Arguments passed to the training
        """
        os.makedirs(args.cached_model_path, exist_ok=True)
        if not os.path.exists(f"{args.cached_model_path}/params.txt"):
            with open(f"{args.cached_model_path}/params.txt", "a") as f:
                self.things_to_write(args, f)
        self.folder_check(args)
        with open(f"{args.pred_path}/{args.experiment_name}/log.txt", "a") as f:
            self.things_to_write(args, f)

    def folder_check(self, args):
        try:
            os.makedirs(f"{args.pred_path}/{args.experiment_name}")
            return
        except OSError:
            print(f"The experiment named {args.experiment_name} already exists! \nCreating a new one named {args.experiment_name}_1")
            args.experiment_name = f"{args.experiment_name}_1"
            self.folder_check(args)

    def things_to_write(self, args, f):
        date_time = datetime.fromtimestamp(time.time()).strftime("%d-%m-%Y, %H:%M:%S")
        f.write(f"Timestamp: {date_time} \n")
        f.write(f"Training: {args.experiment_name} \n")
        f.write(f"Metric: {args.metric} \n")
        f.write(f"Evaluation: {args.evaluate_only} \n")
        f.write(f"World: {args.world} \n")
        f.write(f"Cached Model: {args.cached_model} \n")
        f.write(f"Downsample: {args.downsample} \n")
        f.write(f"True/False threshold: {args.tf_metric} \n")

        f.write("\nArgs")
        for i in inspect.getmembers(args):
            if not i[0].startswith('_'):
                if not inspect.ismethod(i[1]):
                    f.write(f"{str(i)} \n")
        f.write("\nHyperParameters")
        for i in inspect.getmembers(super()):
            if not i[0].startswith('_'):
                if not inspect.ismethod(i[1]):
                    f.write(f"{str(i)} \n")
        f.write("\nModel Config")
        for i in inspect.getmembers(self):
            if not i[0].startswith('_'):
                if not inspect.ismethod(i[1]):
                    f.write(f"{str(i)} \n")
        f.write("\n \n \n")
