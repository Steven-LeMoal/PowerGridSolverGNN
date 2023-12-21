import numpy as np
import json
import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class Dataloader(object):
    def __init__(
        self,
        dire_path,
        lag_step=4,
        lag_jump=10,
        batch_size=64,
        train_percent=0.7,
        shuffle_main=True,
        shuffle_val=False,
        log_edge_attr=True,
        adjacency_file="/adjacency.json",
        injections_file="/injections.npy",
        loads_file="/loads.npy",
    ):
        self.dire_path = dire_path
        self.train_percent = train_percent
        self.batch_size = batch_size

        self.log_edge_attr = log_edge_attr

        self.lag_step = lag_step
        self.lag_jump = lag_jump

        main_graphs, val_graphs = self.create_feature(
            adjacency_file, injections_file, loads_file
        )

        self.main_loader = self.create_dataloader(main_graphs, shuffle=shuffle_main)

        self.val_loader = (
            self.create_dataloader(val_graphs, shuffle=shuffle_val)
            if val_graphs
            else None
        )

    def create_dataloader(self, graphs, shuffle=True):
        return DataLoader(graphs, batch_size=self.batch_size, shuffle=shuffle)

    def create_split(self, graph_data):
        if self.train_percent < 1:
            train_size = int(self.train_percent * len(graph_data))
            train_graphs = graph_data[:train_size]
            val_graphs = graph_data[train_size:]
            return train_graphs, val_graphs
        return graph_data, None

    def create_feature(self, adjacency_file, injections_file, loads_file):
        adjacency_file_path = self.dire_path + adjacency_file

        with open(adjacency_file_path, "r") as file:
            adjacency_data = json.load(file)

        injections_file_path = self.dire_path + injections_file
        loads_file_path = self.dire_path + loads_file

        injections_data = np.load(injections_file_path)
        loads_data = np.load(loads_file_path)

        return self.create_graph_features(adjacency_data, injections_data, loads_data)

    def create_edge_index(self, adjacency_data):
        edge_index = [
            [value["from"], value["to"]] for key, value in adjacency_data.items()
        ]
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    def create_edge_attr(self, adjacency_data):
        edge_attr = [[value["reactance"]] for key, value in adjacency_data.items()]
        return torch.tensor(edge_attr, dtype=torch.float)

    def create_edge_attr_log(self, adjacency_data):
        edge_attr = [
            [torch.log(torch.tensor(value["reactance"], dtype=torch.float) + 1)]
            for key, value in adjacency_data.items()
        ]
        return torch.tensor(edge_attr, dtype=torch.float)

    def create_node_features(self, injections_data, t):
        all_node_features = []
        for node_index in range(injections_data.shape[1]):
            past_injections = []
            current_injection = injections_data[t, node_index]

            for n in range(1, self.lag_step + 1):
                lag_time = t - n * self.lag_jump
                injection_to_add = injections_data[max(lag_time, 0), node_index]

                past_injections.append(
                    injection_to_add
                    if lag_time >= 0
                    else past_injections[-1]
                    if past_injections
                    else current_injection
                )
            past_injections.append(current_injection)
            all_node_features.append(past_injections)

        return torch.tensor(all_node_features, dtype=torch.float)

    def create_graph_features(self, adjacency_data, injections_data, loads_data):
        edge_index = self.create_edge_index(adjacency_data)

        if self.log_edge_attr:
            edge_attr = self.create_edge_attr_log(adjacency_data)
        else:
            edge_attr = self.create_edge_attr(adjacency_data)

        graphs = []
        for t in range(self.lag_step * self.lag_jump, injections_data.shape[0]):
            x = self.create_node_features(injections_data, t)
            y = torch.tensor(loads_data[t, :], dtype=torch.float).view(-1, 1)
            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            graphs.append(graph)
        return self.create_split(graphs)
