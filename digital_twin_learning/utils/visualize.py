# python
import argparse
import matplotlib
import numpy as np
import open3d as o3d
import time
import os
import trimesh
from torch.utils.tensorboard.writer import SummaryWriter
from PIL import Image
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from digital_twin_learning.utils.helpers import PCDHandler
import math
import matplotlib.pyplot as plt
import seaborn as sns
if __name__ != "__main__":
    from .helpers import tensorboard_img
    from .graph import Graph, GraphConfig
else:
    from digital_twin_learning.utils.graph import Graph, GraphConfig


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


MESH_DIRECTORY = "digital_twin_learning/resources/meshes/rand_gen_3"
GRAPH_DIRECTORY1 = "digital_twin_learning/resources/graphs/rand_gen_3/final.graphml.xml"
GRAPH_DIRECTORY2 = "digital_twin_learning/results/predictions/mix3_edge_success_small_repres_5_eval_CoT_1_1_1_1/graph/inferred.graphml.xml"


def read_mesh(directory):
    paths = [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith(".obj")
    ]
    meshes = [trimesh.load(path, force="mesh") for path in paths]
    mesh = trimesh.util.concatenate(meshes)
    return mesh.as_open3d.compute_vertex_normals()


def read_nodes(nodedict):
    points = np.array([node.position for node in nodedict.values()])
    colors = np.array([[1.0, 1.0, 0.0] for node in nodedict.values()])
    nodes = o3d.geometry.PointCloud()
    nodes.points = o3d.utility.Vector3dVector(points)
    nodes.colors = o3d.utility.Vector3dVector(colors)
    return nodes


def read_edges(edgelist, nodes, metric="success", path=""):
    if metric == "CoT":
        lines = np.array([[edge.start_id, edge.goal_id] for edge in edgelist if not math.isnan(edge.cost)])
        maxi = np.nanmax([edge.cost for edge in edgelist])
        mini = np.nanmin([edge.cost for edge in edgelist])
        # sns.displot(data=np.array([(edge.cost - mini)/(maxi-mini) for edge in edgelist]))
        print(maxi)
        print(mini)
        # fig, ax = plt.subplots(figsize=(6, 1))
        # fig.subplots_adjust(bottom=0.5)
        # cmap = matplotlib.cm.Reds
        # norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
        # fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation="horizontal", label="CoT")
        # plt.savefig(f"{path}/CoT_Colormap.png", dpi=300)
        # plt.close()
        # # plt.show()
        colors = np.array([matplotlib.cm.Reds((((edge.cost - mini)/(maxi-mini))))[:3] for edge in edgelist if not math.isnan(edge.cost)])
    else:
        lines = np.array([[edge.start_id, edge.goal_id] for edge in edgelist])
        colors = np.array([matplotlib.cm.autumn(edge.cost)[:3] for edge in edgelist])
    edges = o3d.geometry.LineSet()
    edges.points = nodes.points
    edges.lines = o3d.utility.Vector2iVector(lines)
    edges.colors = o3d.utility.Vector3dVector(colors)
    return edges


def visualize(args, GRAPH_DIRECTORY, interactive: bool = True):
    MESH_DIRECTORY = args.mesh_dir

    config = GraphConfig()
    graph = Graph()

    config.graph_file = GRAPH_DIRECTORY
    graph.initialize(config, metric=args.metric)
    print("Number of nodes:", graph.num_nodes())
    print("Number of edges:", graph.num_edges())
    # SAVE_PATH = f"{args.pred_path}/{args.experiment_name}"
    mesh = read_mesh(MESH_DIRECTORY)
    nodes = read_nodes(graph.node_dict)
    edges = read_edges(graph.edge_list, nodes, args.metric, "SAVE_PATH")
    # o3d.visualization.draw_geometries([mesh, nodes])
    if interactive:
        o3d.visualization.draw_geometries([mesh, nodes, edges])
    return mesh, nodes, edges


def delta_edges(edgelist1, edgelist2, nodes, metric="success"):
    if metric == "CoT":
        lines = np.array([[edge.start_id, edge.goal_id] for edge in edgelist1 if not math.isnan(edge.cost)])
        maxi = np.nanmax([(edge1.cost - edge2.cost) for edge1, edge2 in zip(edgelist1, edgelist2)])
        colors = np.array([matplotlib.cm.summer(abs(edge1.cost - edge2.cost) / maxi)[:3] for edge1, edge2 in zip(edgelist1, edgelist2) if not math.isnan(edge1.cost)])
    else:
        lines = np.array([[edge.start_id, edge.goal_id] for edge in edgelist1])
        colors = np.array([matplotlib.cm.summer(abs(edge1.cost - edge2.cost))[:3] for edge1, edge2 in zip(edgelist1, edgelist2)])
        # mini = np.nanmin([abs(edge1.cost - edge2.cost) for edge1, edge2 in zip(edgelist1, edgelist2)])
        # maxi = np.nanmax([abs(edge1.cost - edge2.cost) for edge1, edge2 in zip(edgelist1, edgelist2)])
        # fig, ax = plt.subplots(figsize=(6, 1))
        # fig.subplots_adjust(bottom=0.5)
        # cmap = matplotlib.cm.summer
        # norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
        # fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation="horizontal", label="CoT")
        # # plt.savefig(f"{path}/CoT_Colormap.png", dpi=300)
        # # plt.close()
        # plt.show()
    edges = o3d.geometry.LineSet()
    edges.points = nodes.points
    edges.lines = o3d.utility.Vector2iVector(lines)
    edges.colors = o3d.utility.Vector3dVector(colors)
    return edges


def visualize_delta(args, GRAPH_DIRECTORY1, GRAPH_DIRECTORY2, interactive: bool = True):
    MESH_DIRECTORY = args.mesh_dir

    config1 = GraphConfig()
    graph1 = Graph()

    config1.graph_file = GRAPH_DIRECTORY1
    graph1.initialize(config1, metric=args.metric)

    config2 = GraphConfig()
    graph2 = Graph()

    config2.graph_file = GRAPH_DIRECTORY2
    graph2.initialize(config2, metric=args.metric)

    print("Number of nodes:", graph1.num_nodes())
    print("Number of edges:", graph1.num_edges())

    mesh = read_mesh(MESH_DIRECTORY)
    nodes = read_nodes(graph1.node_dict)
    edges = delta_edges(graph1.edge_list, graph2.edge_list, nodes, args.metric)
    print(edges)
    if interactive:
        o3d.visualization.draw_geometries([mesh, nodes, edges])
    return mesh, nodes, edges


def tensorboard(edges, mesh, nodes, pointcloud, tb: SummaryWriter, extra=""):
    tb.add_3d(f"Edges {extra}", to_dict_batch([edges]), step=0)
    if mesh:
        tb.add_3d(f"Complete mesh {extra}", to_dict_batch([mesh]), step=0)
        tb.add_3d(f"Nodes {extra}", to_dict_batch([nodes]), step=0)
        tb.add_3d(f"PointCloud {extra}", to_dict_batch([pointcloud]), step=0)


def save_img_pcd(mesh, nodes, edges, args, extra=""):
    # Visualize Point Cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.add_geometry(nodes)
    vis.add_geometry(edges)

    # Updates
    vis.poll_events()
    vis.update_renderer()

    # Capture image
    time.sleep(1)
    vis.capture_screen_image(f"{args.pred_path}/{args.experiment_name}/visualizer{extra}.png", do_render=True)

    # Close
    vis.destroy_window()


def nav_graph_tb_save(args, GRAPH_DIR_GT, GRAPH_DIRECTORY, tb: SummaryWriter):
    """Script to generate a nav graph and save it on Tensorboard"""
    pointcloud = PCDHandler(args.pcd_path)
    if args.downsample > 0.0:
        pointcloud.subsample_pcd(voxel_size=args.downsample)
    mesh, nodes, edges = visualize(args, GRAPH_DIRECTORY, False)
    save_img_pcd(mesh, nodes, edges, args)
    tensorboard_img(f"{args.pred_path}/{args.experiment_name}/visualizer.png", "Navigation Graph", tb)
    tensorboard(edges, mesh, nodes, pointcloud.pcd, tb)
    mesh, nodes, edges = visualize_delta(args, GRAPH_DIR_GT, GRAPH_DIRECTORY, False)
    save_img_pcd(mesh, nodes, edges, args, "_delta")
    tensorboard_img(f"{args.pred_path}/{args.experiment_name}/visualizer_delta.png", "Delta Graph", tb)
    tensorboard(edges, None, None, None, tb, "delta")
    mesh, nodes, edges = visualize(args, GRAPH_DIR_GT, False)
    save_img_pcd(mesh, nodes, edges, args, "_GT")
    tensorboard_img(f"{args.pred_path}/{args.experiment_name}/visualizer_GT.png", "GT Graph", tb)
    tensorboard(edges, None, None, None, tb, "GT")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction_path",
        type=str,
        default="/home/lpiglia/git/digital_twin_navigation/digital_twin_learning/digital_twin_learning/results/predictions/",
    )
    parser.add_argument("--experiment_name", type=str, default="default_ETH_LEE")
    args = parser.parse_args()
    args.mesh_dir = MESH_DIRECTORY
    args.metric = "CoT"

    mesh, nodes, edges = visualize(args, GRAPH_DIRECTORY1)
    mesh, nodes, edges = visualize(args, GRAPH_DIRECTORY2)
    mesh, nodes, edges = visualize_delta(args, GRAPH_DIRECTORY1, GRAPH_DIRECTORY2)
    # save_img_pcd(mesh, nodes, edges, args)

    # mesh = read_mesh(MESH_DIRECTORY)
    # o3d.visualization.draw_geometries([mesh])


# def main():
#     dir = "/home/lpiglia/git/digital_twin_navigation/digital_twin_learning/digital_twin_learning/results/predictions/all_data_no_pose"
#     for filename in os.listdir(dir):
#         f = os.path.join(dir, filename)
#         # checking if it is a file
#         if os.path.isfile(f):
#             print(f)
#             if f == "/home/lpiglia/git/digital_twin_navigation/digital_twin_learning/digital_twin_learning/results/predictions/all_data_no_pose/3DImg_GT_1.0_pred_0.051712602376937866_all_data_no_pose.pcd":
#                 cloud = o3d.io.read_point_cloud(f) # Read the point cloud
#                 o3d.visualization.draw_geometries([cloud]) # Visualize the point cloud

# if __name__ == "__main__":
#     main()
