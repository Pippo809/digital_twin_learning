# python
from dataclasses import dataclass
import networkx as nx
import numpy as np
import math
from typing import Dict, List, Tuple, Union



@dataclass
class GraphConfig:
    graph_file: str = ""
    method: str = "dijkstra"
    metric: str = "euclidean"

    @dataclass
    class thresholds:
        success: float = 0.5

    @dataclass
    class weights:
        success: float = 1.0
        euclidean_distance: float = 1.0
        traversed_distance: float = 1.0
        CoT: float = 1.0


class Graph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def initialize(self, config, new_costs=None, metric="success", sample=False):
        self.metric = metric
        self.config = config
        self.load_graph()
        if new_costs is not None:
            self.new_costs_fun(new_costs)
        self.create_helpers(sample)
        self._to_lists(sample)

    def shutdown(self):
        pass

    def _to_lists(self, sample=False):
        if sample:
            self.data_list = list(self.graph.nodes.data())
            self.success_list = np.array([cost[1][self.metric] for cost in self.data_list])
        else:
            self.data_list = list(self.graph.edges.data())
            self.success_list = np.array([cost[2][self.metric] for cost in self.data_list])

    def load_graph(self):
        self.graph = nx.readwrite.graphml.read_graphml(self.config.graph_file)

    def save(self, directory, stem):
        nx.readwrite.graphml.write_graphml(
            self.graph, directory + stem + ".graphml.xml"
        )

    def num_nodes(self):
        return self.graph.number_of_nodes()

    def num_edges(self):
        return self.graph.number_of_edges()

    def apply_thresholds(self):
        thresholded_edges = [
            (u, v)
            for (u, v, data) in list(self.graph.edges(data=True))
            if self._threshold_function(data)
        ]
        self.graph.remove_edges_from(thresholded_edges)
        self.graph.remove_nodes_from(list(nx.isolates(self.graph)))

    def compute_weights(self):
        weighted_edges = [
            (u, v, self._weight_function(data))
            for (u, v, data) in list(self.graph.edges(data=True))
        ]
        self.graph.add_weighted_edges_from(weighted_edges, weight="weight")

    def create_helpers(self, sample):
        if sample:
            self.node_dict = {
                node[0]: NodeSampling.from_tuple(node) for node in list(self.graph.nodes(data=True))
            }
        else:
            self.edge_list = [
                Edge.from_tuple(edge) for edge in list(self.graph.edges(data=True))
            ]
            self.node_dict = {
                node[0]: Node.from_tuple(node) for node in list(self.graph.nodes(data=True))
            }
        self.pose_array = np.array(
            [
                self._networkx_data_to_numpy_pose(data)
                for (id, data) in list(self.graph.nodes(data=True))
            ]
        )
        self.pose_dict = {
            id: self._networkx_data_to_numpy_pose(data)
            for (id, data) in list(self.graph.nodes(data=True))
        }

    def new_costs_fun(self, costs):
        print(len(list(self.graph.edges.data())))
        print(len(costs))
        k = 0
        for i, (u, v, data) in enumerate(list(self.graph.edges.data())):
            if math.isnan(data[self.metric]):
                pass
            else:
                self.graph.remove_edge(u, v)
                data[self.metric] = costs[k]
                self.graph.add_edge(u, v, **data)
                k += 1

    def _threshold_function(self, data):
        return data["success"] < self.config.thresholds.success

    def _weight_function(self, data):
        weight = (
            self.config.weights.success * (1.0 - data["success"])
            + self.config.weights.euclidean_distance * data["euclidean_distance"]
            + self.config.weights.traversed_distance * data["traversed_distance"]
        )
        return weight

    def _closest_node_to_pose(self, pose):
        # TODO(areske): search only in 2D?
        query = np.array([pose[:3]])
        pool = self.pose_array[:, :3]
        return self.node_list[cdist(query, pool, self.config.metric).argmin()]

    @staticmethod
    def _add_orientation_to_node_poses(path):
        # TODO(areske): compute roll and pitch as well?
        for i in range(1, len(path) - 1, 1):
            dx, dy = path[i][:2] - path[i - 1][:2]
            yaw = math.atan2(dy, dx)
            path[i][3:] = Rotation.from_euler("z", yaw).as_quat()

    @staticmethod
    def _networkx_data_to_numpy_pose(data):
        return np.array([data["px"], data["py"], data["pz"], 0.0, 0.0, 0.0, 1.0])


@dataclass
class Node:
    id: int

    position: np.array
    normal: np.array

    def to_tuple(self):
        return (
            self.id,
            {
                "px": self.position[0],
                "py": self.position[1],
                "pz": self.position[2],
                "nx": self.normal[0],
                "ny": self.normal[1],
                "nz": self.normal[2],
            },
        )

    @classmethod
    def from_tuple(cls, tuple):
        return Node(
            tuple[0],
            np.array([tuple[1]["px"], tuple[1]["py"], tuple[1]["pz"]]),
            np.array([tuple[1]["nx"], tuple[1]["ny"], tuple[1]["nz"]]),
        )


class Edge:
    def __init__(self, start_id: int, goal_id: int, success: float, euclidean_distance: float,
                traversed_distance: float, CoT: Union[float, None] = None, metric: str = "success"):

        self.start_id = start_id
        self.goal_id = goal_id

        self.success = success
        self.euclidean_distance = euclidean_distance
        self.traversed_distance = traversed_distance
        self.CoT = CoT
        self.cost = self.set_metric(metric)

    def set_metric(self, metric):
        if metric == "success":
            return self.success
        elif metric == "traversed_distance":
            return self.traversed_distance
        elif metric == "CoT":
            assert (self.CoT is not None), "CoT is not defined for this graph"
            # if self.CoT == "nan":
            #     return float(10**8)
            # else:
            return self.CoT

    def to_tuple(self):
        return (
            self.start_id,
            self.goal_id,
            {
                "success": self.success,
                "euclidean_distance": self.euclidean_distance,
                "traversed_distance": self.traversed_distance,
                "CoT": self.CoT,
                "cost": self.cost
            },
        )

    @classmethod
    def from_tuple(cls, tuple):
        return Edge(
            tuple[0],
            tuple[1],
            tuple[2]["success"],
            tuple[2]["euclidean_distance"],
            tuple[2]["traversed_distance"],
            tuple[2]["CoT"],
        )
    
class NodeSampling:
    def __init__(self, id: Union[int, str], position: np.ndarray, pose: np.ndarray, success: float, metric: str = "success"):
        self.id = int(id)

        self.position = position
        self.pose = pose

        self.success = success
        self.cost = self.set_metric(metric)

    def set_metric(self, metric):
        if metric == "success":
            return self.success
        else:
            raise TypeError(f"{metric} is not a right metric")

    def to_tuple(self):
        return (
            self.id,
            {
                "px": self.position[0],
                "py": self.position[1],
                "pz": self.position[2],
                "qx": self.pose[0],
                "qy": self.pose[1],
                "qz": self.pose[2],
                "qw": self.pose[3],
                "success": self.success,
                "cost": self.cost
            },
        )

    @classmethod
    def from_tuple(cls, tuple: Tuple[Union[int, str], Dict[str, float]]):
        return NodeSampling(
            tuple[0],
            np.array([tuple[1]["px"], tuple[1]["py"], tuple[1]["pz"]]),
            np.array([tuple[1]["qx"], tuple[1]["qy"], tuple[1]["qz"], tuple[1]["qw"]]),
            tuple[1]["success"],
        )

 

class AdjacencyList:
    def __init__(self, mean):
        self.nodedict = {}
        self.edgelist = []
        self.mean = mean

    def addnode(self, nodeid, pose):
        if nodeid not in self.nodedict:
            self.nodedict[nodeid] = Node(nodeid, pose)

    def addedge(self, startid, goalid, startpose, goalpose, num_iters, cost):
        """
        @param nodeid: from node
        @param goalid: to node
        @param cost: dict(time, dist, energy, success)
        """
        if startid not in self.nodedict:
            self.addnode(startid, startpose)

        if goalid not in self.nodedict:
            self.addnode(goalid, goalpose)

        edge = Edge(
            startid,
            goalid,
            num_iters,
            cost["success"],
            cost["time"],
            cost["dist"],
            cost["energy"],
        )
        self.edgelist.append(edge)

    def dump(self, filename):
        nodelist = [(nid, self.nodedict[nid].pose + self.mean) for nid in self.nodedict]
        edgelist = [e.to_dict() for e in self.edgelist]
        with open(filename, "wb") as handle:
            pickle.dump((nodelist, edgelist), handle, protocol=pickle.HIGHEST_PROTOCOL)

        return self.nodedict, self.edgelist


@dataclass
class CostInfo:
    start_node: Node
    goal_node: Node
    numiters: int

    success_rate: float = 0.0
    num_successes: int = 0
    doneiters: int = 0
    time: float = 0.0
    energy: float = 0.0

    @property
    def done(self):
        return self.doneiters == self.numiters

    def update_iter(self, success=0, time=0.0):
        if self.done():
            raise ValueError("This cost is already full!")

        self.num_successes += success
        self.time += time
        self.doneiters += 1


def parse_task(args, cfg, sim_params):
    terrain = Terrain(
        cfg["env"]["terrain"],
        cfg["env"]["numEnvs"],
        env_spacing=cfg["env"]["envSpacing"],
    )
    try:
        task = eval(args.task)(
            cfg=cfg,
            terrain=terrain,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            device_type=args.device,
            device_id=args.device_id,
            headless=args.headless,
        )
    except NameError as e:
        raise NameError(f"Error creating the task {args.task}: {e}")
    env = VecTaskPython(
        task,
        args.rl_device,
        clip_actions=cfg["env"]["learn"]["clip_actions"],
        clip_observations=cfg["env"]["learn"]["clip_observations"],
    )

    return task, env
