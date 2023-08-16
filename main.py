import numpy as np

from graph import Node, Graph
from utils.graph_utils import get_adj_map_with_edge_list
from utils.visualizer import visualize

# make node list
node_list = []
for i in range(0, 2):
    for j in range(0, 2):
        for k in range(0, 2):
            node = Node(pos=[i, j, k], i=len(node_list))
            node_list.append(node)

node_list.append(Node(pos=[0.5, 1.5, 0.5], i=len(node_list)))
node_list = np.array(node_list)

edge_list = [[0, 1], [0, 2], [0, 4],
             [1, 0], [1, 3], [1, 5],
             [2, 0], [2, 3], [2, 6],
             [3, 1], [3, 2], [3, 7],
             [4, 0], [4, 5], [4, 6],
             [5, 1], [5, 4], [5, 7],
             [6, 2], [6, 4], [6, 7],
             [7, 3], [7, 5], [7, 6],
             [8, 2], [8, 3], [8, 6], [8, 7]]
adj_map = get_adj_map_with_edge_list(node_list=node_list, edge_list=edge_list)

graph = Graph(node_list=node_list, adj_map=adj_map, is_primitive=False)

visualize(graph=graph)