import datetime
import os
import sys

import numpy as np
import open3d as o3d
import torch
import torch.optim as optim
from typing import Union
import re
from scipy.spatial.transform import Rotation as R
import math
import glob

from digital_twin_learning.models.sampling import CVAE, Decoder, Encoder
from open3d.visualization.tensorboard_plugin import summary
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from prettytable import PrettyTable
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from digital_twin_learning.data.pointnet_sampling import PointNetDatasetSampling, PCDCompletePoses
from digital_twin_learning.utils.statistics import build_graph_from_preds, statistics
from digital_twin_learning.utils.sampling_utils import criteria, kl_regularizer
from digital_twin_learning.utils.visualize import nav_graph_tb_save


class TrainUtilsCVAE:
    def __init__(
        self,
        model: torch.nn.Module,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        map_encoder: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        classlossfun: torch.nn.BCEWithLogitsLoss,
        poscountlossfun,
        accuracyfuns: tuple,
        train_loader: DataLoader,
        test_loader: DataLoader,
        tb,
        test_data: PointNetDatasetSampling,
        device=torch.device("cuda:0"),
        finetune=False,
        metric="success",
        kl_weight=1,
        class_weight=1,
        recon_weight=1,
        poscountweight=0,
    ):
        """Class to define all the functions and data useful
        for training

        Args:
            model (torch.nn.Module): CVAE model
            encoder (torch.nn.Module): Encoder Model
            decoder (torch.nn.Module): Decoder Model
            map_encoder (torch.nn.Module): PointCloud Encoder (to get the latent representation)
            optimizer (torch.optim.Optimizer): Optimizer (IE Adam)
            classlossfun (torch.nn.BCEWithLogitsLoss): Classification Loss fun (Binary Cross Entropy)
            poscountlossfun (torch.nn.PoissonNLLLoss): Map Encoder Loss (Poisson)
            train_loader (DataLoader): Training Dataset
            test_loader (DataLoader): Testing Dataset
            tb (SummaryWriter): TensorBoard handle
            test_data (PointNetDataset): Test Data (No Dataloader)
            modifier (RandomPCDChange): Inject Noise/Dropout Points
            device (str, optional): torch device. Defaults to torch.device("cuda:0").
            finetune (bool, optional): Flag to activate finetuning. Defaults to False.
            kl_weight (int, optional): KL Loss weight. Defaults to 1.
            class_weight (int, optional): Classification Loss weight. Defaults to 1.
            recon_weight (int, optional): Reconstruction Loss weight. Defaults to 1.
            poscountweight (int, optional): Poisson Loss weight. Defaults to 0.
        """

        self.model = model
        self.map_encoder = map_encoder
        self.decoder = decoder
        self.encoder = encoder
        self.optimizer = optimizer
        self.classlossfun = classlossfun
        self.poscountlossfun = poscountlossfun
        self.accuracyfuns = accuracyfuns
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.tb = tb
        self.kl_weight = kl_weight
        self.class_weight = class_weight
        self.recon_weight = recon_weight
        self.poscountweight = poscountweight
        self.device = device
        self.test_data = test_data
        self.finetune = finetune
        self.metric = metric
        self.model.to(device)

    def run(self, args):
        """Main method to run the training pipeline
        """
        if args.evaluate_only:

            assert os.path.exists(f"{args.cached_model_path}/cvae_{args.epoch_model}.pt"), f"{args.cached_model_path}/cvae_{args.epoch_model}.pt model not found"
            assert os.path.exists(f"{args.cached_model_path}/map_encoder_{args.epoch_model}.pt"), f"{args.cached_model_path}/map_encoder_{args.epoch_model}.pt model not found"
            assert os.path.exists(f"{args.cached_model_path}/decoder_{args.epoch_model}.pt"), f"{args.cached_model_path}/decoder_{args.epoch_model}.pt model not found"

            print(f"Loading checkpoint: {args.cached_model_path}/cvae_{args.epoch_model}.pt")
            self.model.load_state_dict(torch.load(f"{args.cached_model_path}/cvae_{args.epoch_model}.pt", map_location=args.device))
            self.map_encoder.load_state_dict(torch.load(f"{args.cached_model_path}/map_encoder_{args.epoch_model}.pt", map_location=args.device))
            self.decoder.load_state_dict(torch.load(f"{args.cached_model_path}/decoder_{args.epoch_model}.pt", map_location=args.device))

            preds, scores = self.inference(args)
            self.tb.close()
            os.makedirs(f"{args.data_cachedir}/{args.cached_dataset}_learned{args.stem}", exist_ok=True)
            data = PCDCompletePoses(args.pcd_path, self.test_data, preds, scores, f"{args.data_cachedir}/{args.cached_dataset}_learned{args.stem}/all_with_0_samples")
            
            return preds, scores
    
        if args.cached_encoder_path != "":
            self.finetune = True
            print(f"Loading checkpoint: {args.cached_encoder_path}/{args.epoch_model}.pt")
            self.map_encoder.load_state_dict(torch.load(f"{args.cached_encoder_path}/{args.epoch_model}.pt", map_location=args.device))
            print("Training the model..")
            loss_viz, recon_loss_viz, kl_loss_viz, class_loss_viz, preds = self.train(args)

        elif not os.path.exists(f"{args.cached_model_path}/cvae_{args.epoch_model}.pt"):
            print(args.cached_model_path)
            print("Training the model..")
            loss_viz, recon_loss_viz, kl_loss_viz, class_loss_viz, preds = self.train(args)

        else:
            print("Loading checkpoint")
            self.model.load_state_dict(torch.load(f"{args.cached_model_path}/cvae_{args.epoch_model}.pt", map_location=args.device))
            self.map_encoder.load_state_dict(torch.load(f"{args.cached_model_path}/map_encoder_{args.epoch_model}.pt", map_location=args.device))
            self.decoder.load_state_dict(torch.load(f"{args.cached_model_path}/decoder_{args.epoch_model}.pt", map_location=args.device))

            if args.evaluate_only:
                preds = self.inference(args)
                loss_viz, recon_loss_viz, kl_loss_viz, class_loss_viz, encoder_loss_viz = self.test(0, args)
                message = f"Test loss = {loss_viz}\n\
                            Test Recon Loss = {recon_loss_viz}\n\
                            Test KL Loss = {kl_loss_viz}\n"
                print(message)

            else:
                try:
                    epoch_model = int(args.epoch_model[5:])
                except ValueError:
                    print("If proceeding from a checkpoint you also need to specify an epoch, \n\
                         proceeding to load the latest saved epoch")
                    paths = glob.glob(f"{args.cached_model_path}/*")
                    epoch_model = 0
                    for s in paths:
                        num = max(map(int, re.findall(r'\d+', s)))
                        if num > epoch_model:
                            epoch_model = num
                args.num_epochs = epoch_model + args.num_epochs
                loss_viz, recon_loss_viz, kl_loss_viz, class_loss_viz, preds = self.train(args, resume=epoch_model)

        self.tb.close()

    def train(self, args, resume=0):
        assert args.num_epochs > 0
        loss_viz, recon_loss_viz, kl_loss_viz, class_loss_viz, pred_loss_viz = 0.0, 0.0, 0.0, 0.0, 0.0
        for epoch in range(resume, args.num_epochs):
            try:
                self.model.train()

                Loss, Recon_loss, KL_loss, Class_loss, Pred_loss = self.train_epoch(epoch, args)

                message = f"""After epoch {epoch}: Train loss = {Loss:.3f}, Test loss = {loss_viz:.3f}
Train Recon Loss= {Recon_loss:.3f}, Test Recon Loss = {recon_loss_viz:.3f}
Train KL Loss = {KL_loss:.3f}, Test KL Loss = {kl_loss_viz:.3f}
Train Classification Loss = {Class_loss:.3f}, Test Classification Loss = {class_loss_viz:.3f}
Pred Loss = {Pred_loss:.3f}, Test Pred Loss = {pred_loss_viz:.3f}"""
                print(message)

                if self.finetune:
                    epoch_ = f"{epoch}_finetuned"
                else:
                    epoch_ = epoch
                print(f" ==> saving last model to {args.cached_model_path}")
                torch.save(self.model.state_dict(), f"{args.cached_model_path}/cvae_epoch_{epoch_}.pt")
                torch.save(self.map_encoder.state_dict(), f"{args.cached_model_path}/map_encoder_epoch_{epoch_}.pt")
                torch.save(self.decoder.state_dict(), f"{args.cached_model_path}/decoder_epoch_{epoch_}.pt")

                loss_viz, recon_loss_viz, kl_loss_viz, class_loss_viz, pred_loss_viz = self.test(epoch, args)

            except KeyboardInterrupt:
                print(f" ==> saving last model to {args.cached_model_path}")
                torch.save(self.model.state_dict(), f"{args.cached_model_path}/cvae_epoch_{epoch}.pt")
                torch.save(self.map_encoder.state_dict(), f"{args.cached_model_path}/map_encoder_epoch_{epoch}.pt")
                torch.save(self.decoder.state_dict(), f"{args.cached_model_path}/decoder_epoch_{epoch}.pt")
                preds = self.inference(args)
                return loss_viz, recon_loss_viz, kl_loss_viz, class_loss_viz, preds

        preds = self.inference(args)

        torch.save(self.model.state_dict(), f"{args.cached_model_path}/cvae_final.pt")
        torch.save(self.map_encoder.state_dict(), f"{args.cached_model_path}/map_encoder_final.pt")
        torch.save(self.decoder.state_dict(), f"{args.cached_model_path}/decoder_final.pt")
        return loss_viz, recon_loss_viz, kl_loss_viz, class_loss_viz, preds

    def train_epoch(self, epoch, args):

        model, map_encoder = self.model.train().to(self.device), self.map_encoder.train().to(self.device)

        if self.finetune:
            map_encoder.train(False)
            map_encoder.requires_grad_(False)

        if epoch == 0:
            print("Model parameters:")
            self.count_parameters(model)
            print("Map encoder parameters:")
            self.count_parameters(map_encoder)

        epoch_loss, recon_loss_train, class_loss_train, kl_loss_train, encoder_pred_loss_train = [], [], [], [], []
        tk0 = tqdm(
            enumerate(self.train_loader),
            total=int(len(self.train_loader)),
            desc=f"Training... Epoch {epoch}",
        )

        for batchid, (points, start, pose, success, pos_count, class_label) in tk0:
            self.optimizer.zero_grad()

            # Transform PCD points such that they can be fed to the network
            points = points.data.numpy()
            if args.jitter_train:
                points = self.jitter_points(points)
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            points, start, pose, success, class_label = points.to(self.device), start.to(self.device), pose.to(self.device), success.to(self.device).float(), class_label.float().to(self.device)

            # Use the Map Encoder to get a smaller representation of the PointCloud and a score for the easyness of the voxel
            posefeats = torch.cat((start, start), dim=1).to(self.device).to(points)
            encoder_pred, y_train = map_encoder(points, posefeats)
            y_train, encoder_pred = y_train.float(), encoder_pred.float().to(self.device)
            x_train = torch.cat((start.float(), pose.float()), dim=1)

            # Use the CVAE to get the reconstruction of the pose and the prediction for the success
            mu, logvar, recon_batch, pred = model(x_train, y_train)

            # Get the various losses (reconstruction, regularization and classification)
            recon_loss = criteria(recon_batch, x_train, self.recon_weight)
            kl_loss = kl_regularizer(mu, logvar, self.kl_weight)
            encoder_pred_loss = torch.zeros(recon_loss.shape, device=self.device)  # self.poscountlossfun(torch.squeeze(encoder_pred), class_label)
            classification_loss = self.classlossfun(torch.squeeze(pred).float(), success) * self.class_weight
            loss = torch.mean(recon_loss + kl_loss + classification_loss, 0)
            recon_loss, kl_loss, classification_loss, encoder_pred_loss = torch.mean(recon_loss, 0), torch.mean(kl_loss, 0), torch.mean(classification_loss, 0), torch.mean(encoder_pred_loss, 0)
            # Create a visualization of the network
            if epoch == 0 and batchid == 0:
                self.create_repres(args, (x_train, y_train), (points, posefeats))

            # Backwards propagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            self.optimizer.step()

            # Log the losses
            epoch_loss.append(loss.item())
            kl_loss_train.append(kl_loss.item())
            recon_loss_train.append(recon_loss.item())
            class_loss_train.append(classification_loss.item())
            encoder_pred_loss_train.append(encoder_pred_loss.item())

            # Log the losses on Tensorboard
            self.tb.add_scalar(
                "Train/Loss",
                loss.cpu().item(),
                batchid + len(self.train_loader) * (epoch),
                double_precision=True,
            )
            self.tb.add_scalar(
                "Train/KL loss",
                kl_loss.cpu().item(),
                batchid + len(self.train_loader) * (epoch),
                double_precision=True,
            )
            self.tb.add_scalar(
                "Train/Reconstruction loss",
                recon_loss.cpu().item(),
                batchid + len(self.train_loader) * (epoch),
                double_precision=True,
            )
            self.tb.add_scalar(
                "Train/Classification loss",
                classification_loss.cpu().item(),
                batchid + len(self.train_loader) * (epoch),
                double_precision=True,
            )
            self.tb.add_scalar(
                "Train/Encoder loss",
                encoder_pred_loss.cpu().item(),
                batchid + len(self.train_loader) * (epoch),
                double_precision=True,
            )

            tk0.set_postfix(loss=loss.cpu().item())

        self.tb.add_scalar("Train/AvgLoss", np.mean(np.array(epoch_loss)), epoch)

        return np.mean((np.array(epoch_loss))), np.mean(np.array(recon_loss_train)), np.mean(np.array(kl_loss_train)), np.mean(np.array(class_loss_train)), np.mean(np.array(encoder_pred_loss_train))

    def test(self, epoch: int, args):
        """Function to test and report
        losses on the test dataset

        Parameters
        ----------
        epoch : int
            current epoch

        Returns
        -------
        Losses
        """
        model, map_encoder = self.model.eval(), self.map_encoder.eval()
        model.to(self.device)
        map_encoder.to(self.device)
        with torch.no_grad():
            recon_loss_viz, kl_loss_viz, class_loss_viz, loss_viz, pred_encod_loss_viz = [], [], [], [], []
            tk0 = tqdm(
                enumerate(self.test_loader),
                total=int(len(self.test_loader)),
                desc="Testing... ",
            )

            for batchid, (points, start, pose, success, pos_count, class_label, idx) in tk0:

                points = points.data.numpy()
                if args.jitter_train:
                    points = self.jitter_points(points)
                points = torch.Tensor(points)
                points = points.transpose(2, 1)

                points, success, class_label = points.to(self.device), success.to(self.device), class_label.to(self.device)
                x_val = torch.cat((torch.Tensor(start).float().to(self.device), torch.Tensor(pose).float().to(self.device)), dim=1)

                posefeats = torch.cat((start, start), dim=1).to(self.device).to(points)
                encoder_pred, y_val = map_encoder(points, posefeats)
                y_val = y_val.float()

                mu, logvar, recon_batch, pred = model(x_val, y_val)

                recon_loss = criteria(recon_batch, x_val, self.recon_weight)
                kl_loss = kl_regularizer(mu, logvar, self.kl_weight)
                classification_loss = self.classlossfun(torch.squeeze(pred).float(), success) * self.class_weight
                encoder_pred_loss = self.poscountlossfun(torch.squeeze(encoder_pred).float(), class_label.float().to(self.device))  # self.classlossfun(torch.squeeze(encoder_pred).float(), class_label) * self.poscountweight
                loss = torch.mean(recon_loss + kl_loss, 0)
                recon_loss, kl_loss, classification_loss, encoder_pred_loss = torch.mean(recon_loss, 0), torch.mean(kl_loss, 0), torch.mean(classification_loss, 0), torch.mean(encoder_pred_loss, 0)

                loss_viz.append(loss.item())
                recon_loss_viz.append(recon_loss.item())  # recon_loss.item())
                kl_loss_viz.append(kl_loss.item())  # kl_loss.item())
                class_loss_viz.append(classification_loss.item())
                pred_encod_loss_viz.append(encoder_pred_loss.item())

                tk0.set_postfix(Loss=loss.item(), Recon_loss=recon_loss.item(), Lkl_loss=kl_loss.item())
                # tk0.set_postfix(Loss=loss.item(), Recon_loss=0, Lkl_loss=0, Class_loss=0, encoder_loss=encoder_pred_loss.item())

        self.tb.add_scalar('Test/Reconstruction loss', np.mean(np.array(recon_loss_viz)), epoch)
        self.tb.add_scalar('Test/KL loss', np.mean(np.array(kl_loss_viz)), epoch)
        self.tb.add_scalar('Test/Loss', np.mean(np.array(loss_viz)), epoch)
        self.tb.add_scalar('Test/Classification loss', np.mean(np.array(class_loss_viz)), epoch)
        self.tb.add_scalar('Test/Encoder loss', np.mean(np.array(pred_encod_loss_viz)), epoch)

        return np.mean((np.array(loss_viz))), np.mean(np.array(recon_loss_viz)), np.mean(np.array(kl_loss_viz)), np.mean(np.array(class_loss_viz)), np.mean(np.array(pred_encod_loss_viz))

    def inference(self, args):

        map_encoder = self.map_encoder.eval().to(self.device)
        decoder = self.decoder.eval().to(self.device)

        with torch.no_grad():
            all_pred = []
            all_scores = []
            tk0 = tqdm(
                enumerate(self.test_loader),
                total=int(len(self.test_loader)),
                desc="Evaluating... ",
            )
            for idx, (verts, center, _) in tk0:

                points = verts.data.numpy()
                voxel_pcd = o3d.geometry.PointCloud()
                voxel_pcd.points = o3d.utility.Vector3dVector(verts[0].cpu().detach().numpy())

                points = torch.Tensor(points)
                points = points.transpose(2, 1)

                points = points.to(self.device)
                center = center.to(self.device)
                posefeats = torch.cat((center, center), dim=1).to(self.device).to(points)

                _, y_val = map_encoder(points, posefeats)

                y_val = y_val.float()
                z_val = ((torch.rand((y_val.shape[0], 35)) - 0.5) * 2.2).to(self.device).float()

                x_pred, score = decoder(torch.cat((z_val, y_val), 1))
                
                all_pred.append(torch.tensor(x_pred))
                all_scores.append(score)

                tk0.set_postfix(score=torch.sigmoid(torch.squeeze(score[0])).cpu().item(), x_pred=x_pred[0][:3])
                # print(x_pred[0][:3])
                # print(score)
                # if np.random.random() > 0.95:
                #     o3d.visualization.draw_geometries([voxel_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(origin=x_pred[0][:3].cpu().detach().numpy())])

        all_pred = torch.cat(all_pred, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        return all_pred, all_scores


    def inference_old(self, args, num_samples=300, rand_search_good_point=False):

        plot_3D_path = f"{args.pred_path}/{args.experiment_name}/Plots3D/"
        os.makedirs(plot_3D_path, exist_ok=True)

        map_encoder, decoder = self.map_encoder.eval().to(self.device), self.decoder.eval().to(self.device)
        n = 0
        with torch.no_grad():
            all_pred = []
            all_pred_score = []
            tk0 = tqdm(
                enumerate(self.test_loader),
                total=int(len(self.test_loader)),
                desc="Evaluating... ",
            )

            for batchid, (points, start, pose, success, pos_count, class_label, idx) in tk0:

                points = points.data.numpy()
                points = torch.Tensor(points)
                points = points.transpose(2, 1)

                points, start, success = points.to(self.device), start.to(self.device), success.to(self.device)

                posefeats = torch.cat((start, start), dim=1).to(self.device).to(points)
                pred, y_val = map_encoder(points, posefeats)
                pred = torch.sigmoid(pred)
                # pred = self.from_log_rate_to_prob(pred, l)

                if pos_count[0] > 0:
                    y_val = y_val.float()

                    if rand_search_good_point:  # Deprecated
                        # Try until a good point is found
                        max_pred_score_0 = 0
                        max_pred_score = torch.zeros(16, 1)
                        max_x_pred = torch.zeros(16, 3)
                        k = 0
                        while max_pred_score_0 < 0.9 and k < 1000:
                            z_val = ((torch.rand((y_val.shape[0], 25)) - 0.5) * 2.2).to(self.device).float()
                            x_pred, pred_score = decoder(torch.cat((z_val, y_val), 1))
                            pred_score = torch.sigmoid(pred_score)
                            if pred_score[0] > max_pred_score_0:
                                max_pred_score_0 = pred_score[0]
                                max_pred_score = pred_score
                                max_x_pred = x_pred
                            k += 1
                    else:
                        z_val = ((torch.rand((y_val.shape[0], 35)) - 0.5) * 1.5).to(self.device).float()
                        max_x_pred, pred_score = decoder(torch.cat((z_val, y_val), 1))
                        max_pred_score = torch.sigmoid(pred_score)
                    # print(max_x_pred[0][:])
                    all_pred.append(torch.tensor(max_x_pred))
                    all_pred_score.append(torch.tensor(max_pred_score))

                    if n < num_samples and np.random.random() > 0.95:
                        n += 1
                        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.cpu().numpy()[0].T[:, :3]))
                        pcd.paint_uniform_color([1, 0, 0])
                        self.tb.add_3d("PCD", to_dict_batch([pcd]), step=n)
                        r_orig = R.from_quat(pose[0].cpu().numpy().flatten()).as_matrix()
                        r_pred = R.from_quat(max_x_pred[0, 3:].cpu().numpy().flatten()).as_matrix()
                        mesh1 = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=start[0].cpu().numpy().flatten())
                        mesh1.rotate(r_orig)
                        self.tb.add_3d("Original Sample", to_dict_batch([mesh1]), step=n)
                        mesh2 = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=max_x_pred[0, :3].cpu().numpy().flatten())
                        mesh2.rotate(r_pred)
                        self.tb.add_3d("Generated Sample", to_dict_batch([mesh2]), step=n)
                        self.tb.add_text("Predictions", f"idx {n}th: Predicted success cost: {max_pred_score[0].cpu().item():.3f}; Gt success cost: {success[0].cpu().item():.3f}; GT Samples Number: {pos_count[0].cpu().item():.3f}; Map Encoder out: {pred[0].cpu().item():.3f}", global_step=n)

        return torch.cat(all_pred, dim=0)

    def create_repres(self, args, input_CVAE, input_MAP):
        """Generates a onnx view of the CVAE network
        """
        input_names_CVAE = ["GT 3D Points", "Conditioning"]
        output_names_CVAE = ["mu", "log var", "Reconstructed Point", "Score"]
        input_names_MAP = ["PointCloud"]
        output_names_MAP = ["Score", "Conditioning"]
        torch.onnx.export(self.model, input_CVAE, f"{args.cached_model_path}/CVAE.onnx", input_names=input_names_CVAE, output_names=output_names_CVAE)
        torch.onnx.export(self.map_encoder, input_MAP, f"{args.cached_model_path}/MAP.onnx", input_names=input_names_MAP, output_names=output_names_MAP)

    @staticmethod
    def count_parameters(model):
        """Counts the trainable and freezed parameters of the model
        """
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

    @staticmethod
    def from_log_rate_to_prob(predictions, n):
        """Converts a log rate to a probability (to use with the Poisson Loss)

        Args:
            predictions (tensor): output of the network
            n (int): max number of classes

        Returns:
            tensor: probability of having a count greater than or equal to n
        """
        # Convert the log rate to a rate
        rate = torch.exp(predictions)

        # Calculate the cumulative probability
        cumulative_prob = torch.tensor([math.exp(-rate[i]) * sum([rate[i]**k / math.factorial(k) for k in range(n)]) for i in range(rate.shape[0])], dtype=torch.float)

        # calculate the probability of having a count greater than or equal to n
        prob_greater_equal_n = 1 - cumulative_prob

        return prob_greater_equal_n
