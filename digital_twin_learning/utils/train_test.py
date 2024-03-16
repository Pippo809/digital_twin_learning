import os

import numpy as np
import open3d as o3d
import torch
from open3d.visualization.tensorboard_plugin import summary
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from digital_twin_learning.data.pointnet import RandomPCDChange
from digital_twin_learning.utils.statistics import build_graph_from_preds, statistics
from digital_twin_learning.utils.visualize import nav_graph_tb_save
from digital_twin_learning.utils.graph import Graph, Node, Edge
import networkx as nx


class TrainUtils:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer,
        scheduler,
        lossfun,
        accuracyfuns: tuple,
        train_loader: DataLoader,
        test_loader: DataLoader,
        tb: SummaryWriter,
        test_data,
        modifier: RandomPCDChange,
        device=torch.device("cuda:0"),
        finetune=False,
        metric="success",
        learned="False"
    ):
        """Class to define all the functions and data useful
        for training

        Args:
            model (torch.nn.Module): model of the architecture
            optimizer (_type_): optimizing fun
            scheduler (_type_): scheduling fun
            lossfun (_type_): loss fun
            accuracyfuns (tuple): accurcy functions
            train_loader (DataLoader): training dataset
            test_loader (DataLoader): testing dataset
            tb (SummaryWriter): tensorboard class
            test_data (_type_): test data dataclass
            modifier (RandomPCDChange): Modifing function for jittering dropping PCDs
            device (_type_, optional): Cpu/Cuda,. Defaults to torch.device("cuda:0").
            finetune (bool, optional): finetune/complete train. Defaults to False.
            metric (str, optional): CoT or edge regression. Defaults to "success".
            learned (str, optional): Wether the dataset poses are GT or generated. Defaults to "False".
        """

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lossfun = lossfun
        self.accuracyfuns = accuracyfuns
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.tb = tb
        self.device = device
        self.test_data = test_data
        self.finetune = finetune
        self.metric = metric
        self.model.to(device)
        self.modifier = modifier
        self.learned = learned

    def run(self, args):
        """Main method to run the training pipeline
        """
        if self.learned:
            if not os.path.exists(f"{args.cached_model_path}/{args.epoch_model}.pt"):
                raise ValueError(f"{args.cached_model_path}/{args.epoch_model}.pt not found")
            else:
                print("Loading checkpoint for the sampling map generation")
                self.model.load_state_dict(torch.load(f"{args.cached_model_path}/{args.epoch_model}.pt", map_location=args.device))
                if args.evaluate_only:
                    predictions, labels, generated_graph = self.test_sampling(args.num_epochs, args, sampling=True)
                    graph_dir = build_graph_from_preds(args, generated_graph, predictions)
                    nav_graph_tb_save(args, GRAPH_DIR_GT=generated_graph, GRAPH_DIRECTORY=graph_dir, tb=self.tb)
                    return
            
        if not os.path.exists(f"{args.cached_model_path}/{args.epoch_model}.pt"):
            print(args.cached_model_path)
            print("Training the model..")
            test_loss, test_acc, predictions = self.train(args)
            conf_mat = self.test_and_see(args, 81)
            statistics(args, predictions, test_loss, test_acc, conf_mat, self.tb)

        elif os.path.exists(f"{args.pred_path}/{args.experiment_name}/preds.pt"):
            print(f"Loading results from {args.pred_path}/{args.experiment_name}/predictions.npy")
            predictions = torch.load(f"{args.pred_path}/{args.experiment_name}/preds.pt").cpu().detach().numpy()
            labels = torch.load(f"{args.pred_path}/{args.experiment_name}/labels.pt").cpu().numpy().ravel()
            test_loss, test_acc, predictions = self.test_results(predictions, labels)
            conf_mat = self.test_and_see(args, 81)
            statistics(args, predictions, test_loss, test_acc, conf_mat, self.tb)
            graph_dir = build_graph_from_preds(args, gt_graph=args.cost_path, preds=predictions[0])
            nav_graph_tb_save(args, GRAPH_DIR_GT=args.cost_path, GRAPH_DIRECTORY=graph_dir, tb=self.tb)

        else:
            print("Loading checkpoint")
            self.model.load_state_dict(torch.load(f"{args.cached_model_path}/{args.epoch_model}.pt", map_location=args.device))

            if args.evaluate_only:
                test_loss, test_acc, predictions = self.test(args.num_epochs, args)
                conf_mat = self.test_and_see(args, 81)
                statistics(args, predictions, test_loss, test_acc, conf_mat, self.tb)
                graph_dir = build_graph_from_preds(args, args.cost_path, predictions[0])
                nav_graph_tb_save(args, GRAPH_DIR_GT=args.cost_path, GRAPH_DIRECTORY=graph_dir, tb=self.tb)

            else:
                test_loss, test_acc, predictions = self.train(args)
                conf_mat = self.test_and_see(args, 81)
                statistics(args, predictions, test_loss, test_acc, conf_mat, self.tb)
        self.tb.close()

    def train(self, args):
        assert args.num_epochs > 0
        for epoch in range(args.num_epochs):
            # epoch += 1  # Monkey patch
            self.model.train()
            if epoch > 0:
                self.scheduler.step()

            train_loss, train_acc = self.train_epoch(epoch, args)
            test_loss, test_acc, preds = self.test(epoch, args)
            train_acc0, train_acc1 = train_acc
            test_acc0, test_acc1 = test_acc
            message = f"After epoch {epoch}: train loss = {train_loss:.3f}, test loss = {test_loss:.3f}\n\
train RMLSE = {train_acc0:.3f}, test RMLSE = {test_acc0:.3f}\n\
train L1 loss = {train_acc1:.3f}, test L1 loss = {test_acc1:.3f}\n"
            print(message)

            print(f" ==> saving last model to {args.cached_model_path}")
            if self.finetune:
                epoch = f"{epoch}_finetuned"
            torch.save(self.model.state_dict(), f"{args.cached_model_path}/epoch_{epoch}.pt")
        torch.save(self.model.state_dict(), f"{args.cached_model_path}/final.pt")
        return test_loss, test_acc, preds

    def train_epoch(self, epoch, args):

        model = self.model
        model.train()

        if self.finetune:
            for idx, (name, parameter) in enumerate(model.named_parameters()):
                if idx < 15:
                    parameter.requires_grad_(False)

        if epoch == 0:
            self.count_parameters(model)

        all_preds = []
        all_labels = []
        tk0 = tqdm(
            enumerate(self.train_loader),
            total=int(len(self.train_loader)),
            desc=f"Training... Epoch {epoch}",
        )

        for batchid, (points, labels, start, end) in tk0:
            self.optimizer.zero_grad()

            points = points.data.numpy()

            if args.jitter_train:
                points = self.jitter_points(points)

            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            points = points.to(self.device)
            labels = labels.to(self.device)

            posefeats = torch.cat((start, end), dim=1).to(self.device).to(points)

            preds, _ = model(points, posefeats)
            loss = self.lossfun(preds, labels)

            loss.backward()
            self.optimizer.step()

            all_preds += list(preds.detach().cpu().numpy().ravel())
            all_labels += list(labels.detach().cpu().numpy().ravel())

            self.tb.add_scalar(
                "Train/Loss",
                loss.cpu().item(),
                batchid + len(self.train_loader) * (epoch),
                double_precision=True,
            )

            tk0.set_postfix(loss=loss.cpu().item())

        if args.metric == "success":
            all_preds_ = torch.sigmoid(torch.tensor(all_preds))
        else:
            all_preds_ = torch.tensor(all_preds)
        meanacc = (fun(all_preds_, torch.tensor(all_labels)) for fun in self.accuracyfuns)

        meanloss = self.lossfun(all_preds_, torch.tensor(all_labels))
        acc0, acc1 = (a.item() for a in meanacc)
        self.tb.add_scalar("Train/AvgLoss", meanloss, epoch)
        self.tb.add_scalar("Train/RMSE", acc0, epoch)
        self.tb.add_scalar("Train/L1 Loss", acc1, epoch)

        # for name, weight in model.named_parameters():
        #     if "conv" in name or "linear" in name:
        #         self.tb.add_histogram(name, weight, epoch)

        return meanloss, (acc0, acc1)

    def test(self, epoch: int, args):
        """Function to test and report L1, RMSE and binary cross entropy
        losses on the test dataset

        Parameters
        ----------
        self : TrainUtils
            class with the function for training
        epoch : int
            current epoch

        Returns
        -------
        mean_loss
        (L1, RMSE)
        [Predictions, GT]
        """
        model = self.model.eval()
        model.to(self.device)
        with torch.no_grad():
            # for idx, cost in enumerate(self.test_loader):
            #     points, labels, start, end, ids, _ = self.test_data[idx]
            #     points = points.to(self.device)[None, ...]
            #     labels = labels.to(self.device)[None, ...]
            #     start = torch.tensor(start)[None, ...].to(self.device)
            #     end = torch.tensor(end)[None, ...].to(self.device)

            #     points = points.transpose(2, 1)
            #     posefeats = torch.cat((start, end), dim=1).to(points)

            #     # Apply the model
            #     preds = (self.model(points, posefeats))
            #     max_CoT = np.nanmax((max_CoT, preds.cpu().flatten().numpy()))

            testloss = []
            all_preds = []
            all_preds_ = []
            all_labels = []
            tk0 = tqdm(
                enumerate(self.test_loader),
                total=int(len(self.test_loader)),
                desc="Testing... ",
            )

            for batchid, (points, labels, start, end, ids) in tk0:

                points = points.data.numpy()

                if args.jitter_test:
                    points = self.jitter_points(points)

                points = torch.Tensor(points)
                points = points.transpose(2, 1)

                points = points.to(self.device)
                labels = labels.to(self.device)

                posefeats = torch.cat((start, end), dim=1).to(self.device).to(points)

                preds, _ = model(points, posefeats)

                # track loss and accuracy
                loss = self.lossfun(preds.squeeze(), labels.squeeze())

                if args.metric == "success":
                    all_preds_ = torch.sigmoid(torch.tensor(all_preds))
                else:
                    all_preds_ = torch.tensor(all_preds)
                acc = (fun(all_preds_, torch.tensor(all_labels)) for fun in self.accuracyfuns)

                this_num = labels.shape[0]
                curr_loss = loss.cpu().item() / this_num
                testloss.append(curr_loss)
                acc1, acc2 = (a.item() for a in acc)

                all_preds += list(preds.cpu().numpy().ravel())
                all_labels += list(labels.cpu().numpy().ravel())
                tk0.set_postfix(Entropy=curr_loss, RMSE=acc1, L1=acc2)

        if args.metric == "success":
            all_preds_ = torch.sigmoid(torch.tensor(all_preds))
        else:
            all_preds_ = torch.tensor(all_preds)
        meanacc = (fun(all_preds_, torch.tensor(all_labels))for fun in self.accuracyfuns)
        meanloss = self.lossfun(all_preds_, torch.tensor(all_labels))
        acc0, acc1 = meanacc

        self.tb.add_scalar("Test/Loss", meanloss, epoch)
        self.tb.add_scalar("Test/RMSE", acc0, epoch)
        self.tb.add_scalar("Test/L1 Loss", acc1, epoch)

        return (
            meanloss,
            (acc0, acc1),
            np.array([all_preds_.clone().detach().numpy(), all_labels]),
        )

    def all_labels_fun(self):
        with torch.no_grad():
            all_labels = []
            for batchid, (points, labels, start, end, ids) in enumerate(self.test_loader):
                labels = labels.to(self.device)
                all_labels += list(labels.cpu().numpy().ravel())
        return all_labels

    def test_results(self, preds, labels):
        with torch.no_grad():
            meanloss = self.lossfun(torch.tensor(preds), torch.tensor(labels))
            meanacc = (
                fun(torch.sigmoid(torch.tensor(preds)), torch.tensor(labels))
                for fun in self.accuracyfuns
            )
            acc0, acc1 = meanacc

            return (
                meanloss,
                (acc0, acc1),
                np.array([torch.sigmoid(torch.tensor(preds)).numpy(), labels]),
            )

    def test_and_see(self, args, num_sampl=100):
        """Script to evaluate and plot all the samples in 3D

        Parameters
        ----------
        args : Argparse
            Argument class
        model : torch.nn.Module
            The model used for generate the costs
        device : torch.device
            Device to use for eval (cuda/cpu)
        tb : SummaryWriter
            Tensorboard interface
        num_sampl : int, optional
            Maximum number of samples to plot, by default 100
        """

        self.model.eval()
        n = 0
        plot_3D_path = f"{args.pred_path}/{args.experiment_name}/Plots3D/"
        os.makedirs(plot_3D_path, exist_ok=True)
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        errors = 0
        tot = 0

        with torch.no_grad():
            for idx, cost in enumerate(self.test_loader):
                tot += 1

                points, labels, start, end, ids = self.test_data[idx]
                points = points.to(self.device)[None, ...]
                # labels = labels.to(self.device)[None, ...]
                start = torch.tensor(start)[None, ...].to(self.device)
                end = torch.tensor(end)[None, ...].to(self.device)

                points = points.data.cpu().numpy()

                if args.jitter_test:
                    points = self.jitter_points(points)

                points = torch.Tensor(points)

                points = points.transpose(2, 1)
                posefeats = torch.cat((start, end), dim=1).to(points).to(self.device)

                points = points.to(self.device)
                labels = torch.tensor(labels)
                labels = labels.to(self.device)

                # Apply the model
                preds, _ = self.model(points, posefeats)
                if args.metric == "success":
                    preds = torch.sigmoid(preds)

                if n < num_sampl and not idx % 100:
                    point_line = [
                        start.cpu().numpy().flatten(),
                        end.cpu().numpy().flatten(),
                    ]
                    lines = [[0, 1]]
                    colors = [[0.7, 0, 0] for i in range(len(lines))]
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(point_line)
                    line_set.lines = o3d.utility.Vector2iVector(lines)
                    line_set.colors = o3d.utility.Vector3dVector(colors)
                    pcd = o3d.geometry.PointCloud(
                        o3d.utility.Vector3dVector(points.cpu().numpy()[0].T[:, :3])
                    )
                    pcd.paint_uniform_color([1, 0, 0])
                    self.tb.add_3d("PCD", to_dict_batch([pcd]), step=n)
                    mesh1 = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        origin=start.cpu().numpy().flatten()
                    )
                    self.tb.add_3d("Start", to_dict_batch([mesh1]), step=n)
                    mesh2 = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        origin=end.cpu().numpy().flatten()
                    )
                    self.tb.add_3d("Goal", to_dict_batch([mesh2]), step=n)
                    self.tb.add_3d("Path", to_dict_batch([line_set]), step=n)
                    self.tb.add_text(
                        "Predictions",
                        f"idx {n}th: Predicted success cost: {preds.cpu().item()}; Gt success cost: {labels.cpu().item()}",
                        global_step=n,
                    )
                    n += 1

                if abs(preds.cpu().item() - labels.cpu().item()) > args.tf_metric:
                    errors += 1
                    if labels.cpu().item() > 0.5:
                        FN += 1
                    else:
                        FP += 1
                    # print(f"Predicted success cost {preds.cpu().item()}, Gt success cost {labels.cpu().item()}")
                    point_line = [start.cpu().numpy().flatten(), end.cpu().numpy().flatten()]
                    lines = [[0, 1]]
                    colors = [[0.7, 0, 0] for i in range(len(lines))]
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(point_line)
                    line_set.lines = o3d.utility.Vector2iVector(lines)
                    line_set.colors = o3d.utility.Vector3dVector(colors)
                    pcd = o3d.geometry.PointCloud(
                        o3d.utility.Vector3dVector(points.cpu().numpy()[0].T[:, :3])
                    )
                    pcd.paint_uniform_color([1, 0, 0])
                    if n < num_sampl:
                        self.tb.add_3d("PCD", to_dict_batch([pcd]), step=n)
                        mesh1 = o3d.geometry.TriangleMesh.create_coordinate_frame(
                            origin=start.cpu().numpy().flatten()
                        )
                        self.tb.add_3d("Start", to_dict_batch([mesh1]), step=n)
                        mesh2 = o3d.geometry.TriangleMesh.create_coordinate_frame(
                            origin=end.cpu().numpy().flatten()
                        )
                        self.tb.add_3d("Goal", to_dict_batch([mesh2]), step=n)
                        self.tb.add_3d("Path", to_dict_batch([line_set]), step=n)
                        self.tb.add_text(
                            "Errors",
                            f"idx {n}th: Predicted success cost: {preds.cpu().item()}; Gt success cost: {labels.cpu().item()}",
                            global_step=n,
                        )
                        n += 1

                    o3d.io.write_point_cloud(
                        f"{plot_3D_path}3DImg_GT_{labels.cpu().item()}_pred_{preds.cpu().item()}_{args.experiment_name}.pcd",
                        pcd,
                    )
                else:
                    if labels.cpu().item() > 0.5:
                        TP += 1
                    else:
                        TN += 1
        return TP, TN, FP, FN
    
    def test_sampling(self, epoch: int, args, sampling=False):
        
        model = self.model.eval()
        model.to(self.device)

        G = nx.DiGraph()
        with torch.no_grad():
            all_preds_ = []
            all_labels = []
            tk0 = tqdm(
                enumerate(self.test_loader),
                total=int(len(self.test_loader)),
                desc="Testing... ",
            )
            n = 0
            for idx, cost in tk0:

                (points, labels, start, end, center_start, center_goal, score_start, score_goal, ids) = self.test_data[idx]
                points = points[None, ...].to(self.device)
                start = torch.tensor(start)[None, ...].to(self.device)
                end = torch.tensor(end)[None, ...].to(self.device)
                points = points.cpu().numpy()

                if args.jitter_test:
                    points = self.jitter_points(points)

                points = torch.Tensor(points)
                points = points.transpose(2, 1)

                points = points.to(self.device)
                # labels = labels.to(self.device)

                posefeats = torch.cat((start, end), dim=1).to(self.device).to(points)

                preds, _ = model(points, posefeats)

                if args.metric == "success":
                    all_preds_.append(torch.sigmoid(preds).cpu().item())
                    all_labels.append(0)
                else:
                    all_preds_.append(preds.cpu().item())
                    all_labels.append(0)

                if sampling:
                    node1 = Node(n, np.array(center_start), np.array([0, 0, 1])) #, score_start)
                    node2 = Node(n + 1, np.array(center_goal), np.array([0, 0, 1])) #, score_goal)
                    G.add_node(node1.id, **node1.to_tuple()[1])
                    G.add_node(node2.id, **node2.to_tuple()[1])
                    edge = Edge(node1.id, node2.id, success=0.0, euclidean_distance=0.0, traversed_distance=0.0, CoT=0.0)
                    G.add_edge(node1.id, node2.id, **edge.to_tuple()[2])
                    n += 2
                tk0.set_postfix(pred=all_preds_[-1])

        if sampling:
            nx.readwrite.graphml.write_graphml(G, f"{args.cost_path}_sampled.graphml.xml")
            new_graph = f"{args.cost_path}_sampled.graphml.xml"

        return all_preds_, all_labels, new_graph

    def jitter_points(self, points):
        points[:, :, 0:3] = self.modifier.jit_drop(points[:, :, 0:3])
        return points

    @staticmethod
    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters", "Trainable Parameters"])
        total_params = 0
        total_train_params = 0
        for name, parameter in model.named_parameters():
            params = parameter.numel()
            if parameter.requires_grad:
                train_params = parameter.numel()
            else:
                train_params = 0
            table.add_row([name, params, train_params])
            total_params += params
            total_train_params += train_params
        print(table)
        print(f"Total Params: {total_params}")
        print(f"Total Trainable Params: {total_train_params}")
