import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import scipy
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard.writer import SummaryWriter
from open3d.visualization.tensorboard_plugin import summary
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from PIL import Image
from tabulate import tabulate
from torch.utils.data import DataLoader

from .visualize import save_img_pcd
from .helpers import tensorboard_img, copy_folder
from .graph import CostInfo, Graph, GraphConfig


def build_graph_from_preds(args: argparse.Namespace, gt_graph: str, preds: list):
    """Script to generate a graph from a set of predictions

    Parameters
    ----------
    args : argparse.Namespace
        Arguments for paths
    gt_graph : str
        Path to the ground truth navigation graph
    preds : list
        List of the predicted costs for the graph edges

    Returns
    -------
    str
        Path to the generated graph.
    """

    config = GraphConfig()
    new_graph = Graph()

    config.graph_file = gt_graph
    new_graph.initialize(config, new_costs=preds, metric=args.metric)
    PATH = f"{args.pred_path}/{args.experiment_name}/graph"
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    new_graph.save(PATH + "/", "inferred")
    return PATH + "/inferred.graphml.xml"


def plot(args, preds, labels, tb: SummaryWriter):
    """Script to plot and save a kde plot with the ground truth and the
    predicted edge costs labels

    Parameters
    ----------
    args : argparse.Namespace
        Arguments for paths
    preds, labels : lists
        Lists of the predicted and ground truth costs for the graph edges
    tb : SummaryWriter
        Utility for tensorboard reporting
    """

    unique = np.unique(labels)
    out = {}
    for idx, prediction in enumerate(preds):
        if labels[idx] in out:
            out[labels[idx]].append(prediction)
        else:
            out[labels[idx]] = [prediction]
    sns.set_style("whitegrid")

    PATH = f"{args.pred_path}/{args.experiment_name}/plots"
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    p1 = sns.kdeplot(out[unique[0]], bw_adjust=0.5)  # type: ignore
    plt.savefig(PATH + "/pred_0.png", dpi=300)
    plt.close()
    tensorboard_img(PATH + "/pred_0.png", "Predictions/0", tb)

    p2 = sns.kdeplot(out[unique[1]], bw_adjust=0.5)  # type: ignore
    plt.savefig(PATH + "/pred_0.3.png", dpi=300)
    plt.close()
    tensorboard_img(PATH + "/pred_0.3.png", "Predictions/0.3", tb)

    p3 = sns.kdeplot(out[unique[2]], bw_adjust=0.5)  # type: ignore
    plt.savefig(PATH + "/pred_0.6.png", dpi=300)
    plt.close()
    tensorboard_img(PATH + "/pred_0.6.png", "Predictions/0.6", tb)

    p4 = sns.kdeplot(out[unique[3]], bw_adjust=0.5)  # type: ignore
    plt.savefig(PATH + "/pred_1.png", dpi=300)
    plt.close()
    tensorboard_img(PATH + "/pred_1.png", "Predictions/1", tb)
    return
    # p.save(args.prediction_path + f"experiment_name/Predictions_{args.experiment_name}.png")


def statistics(args, predictions, test_loss, test_acc, conf_mat, tb: SummaryWriter):
    """Generates a statistics file

    Args:
        args (argparse): arguments file
        predictions (list): list containing the predicted costs
        test_loss (float): binary cross entropy lost
        test_acc (tuple): L1 and RMSE loss calculated on the test dataset
        tb (SummaryWriter): Tensorboard class to report to Tensorboard
    """
    _, _, r, _, _ = scipy.stats.linregress(predictions[0], predictions[1])
    test_acc0, test_acc1 = test_acc
    TP, TN, FP, FN = conf_mat
    TPR = (TP) / (TP + FN + 10**(-5))  # Recall
    PPV = (TP) / (TP + FP + 10**(-5))  # Precision
    TNR = (TN) / (TN + FP + 10**(-5))  # Specificity
    NPV = (TN) / (TN + FN + 10**(-5))  # Negative Predictin Value
    print(
        f"Test loss = {test_loss:.3f}, test RMSE = {test_acc0:.3f}, test L1 loss = {test_acc1:.3f}, R = {r:.4f}, R squared = {r*r:.4f} \n"
    )
    file_exists = os.path.isfile(f"{args.pred_path}/results.csv")

    with open(f"{args.pred_path}/results.csv", "a") as f:
        csv_writer = csv.writer(f)
        if not file_exists:
            first_row = ["Experiment Name", "Cross Entropy Loss", "RMSE", "L1 Loss", "R", "R^2"]
            csv_writer.writerow(first_row)

        row = [
            f"{args.experiment_name}",
            test_loss.item(),
            test_acc0.item(),
            test_acc1.item(),
            r,
            r * r,
            PPV,
            TPR,
            TNR,
            NPV,
            TP,
            TN,
            FP,
            FN
        ]
        csv_writer.writerow(row)
    with open(f"{args.pred_path}/{args.experiment_name}/statistics.csv", "a") as f:
        csv_writer = csv.writer(f)
        first_row = ["Experiment Name", "Cross Entropy Loss", "RMSE", "L1 Loss", "R", "R^2", "PPV", "TPR", "TNR", "NPV", "TP", "TN", "FP", "FP"]
        csv_writer.writerow(first_row)
        row = [
            f"{args.experiment_name}",
            test_loss.item(),
            test_acc0.item(),
            test_acc1.item(),
            r,
            r * r,
            PPV,
            TPR,
            TNR,
            NPV,
            TP,
            TN,
            FP,
            FN
        ]
        csv_writer.writerow(row)
    plot(args, predictions[0], predictions[1], tb)
    if args.save_pred:
        with open(f"{args.pred_path}/{args.experiment_name}/predictions.npy", "wb") as f:
            np.save(f, predictions)
    return
