# gt, labeled, unlabeled
import random

import vpython.no_notebook
from vpython import *
import vpython as vp
import numpy as np

def visualize_graph(graph, center_pos):
    node_count = len(graph.node_list)
    edge_count = len(graph.adj_map[np.where(graph.adj_map == 1)]) // 2

    node = [sphere(radius=0.02, color=color.blue) for _ in range(node_count)]
    edge = [cylinder(radius=0.01, color=color.red) for _ in range(0, edge_count)]

    for i, graph_node in enumerate(graph.node_list):
        center_node_pos = graph.node_list[0].node_pos * 1.5 + center_pos
        node[i].pos = vector(*graph_node.node_pos + center_node_pos)

        if graph_node.is_center:
            node[i].color = color.black

    edge_index = 0
    for i, graph_node in enumerate(graph.node_list):
        center_node_pos = graph.node_list[0].node_pos * 1.5 + center_pos

        adj_list = graph.adj_map[graph_node.node_index]
        adj_list = adj_list[:graph_node.node_index + 1]
        neighbor_node_list = graph.node_list[np.where(adj_list == 1)]

        for neighbor_node in neighbor_node_list:
            edge[edge_index].pos = vector(*graph_node.node_pos + center_node_pos)
            edge[edge_index].axis = vector(*neighbor_node.node_pos + center_node_pos) - vector(
                *graph_node.node_pos + center_node_pos)

            edge_index += 1

def visualize(graph):
    window = canvas(x=0, y=0, width=1200, height=1200, center=vector(0, 0, 0), background=vector(0.85, 0.85, 0.85))

    idx = 0
    while True:
        # for primitive_graph in graph.primitive_graph_list:
        #     visualize_graph(graph=primitive_graph, center_pos=[3, 3, 3])

        for hierarchy_graph in graph.hierarchy_graph_map[2]:
            if idx > 200:
                continue

            idx += 1
            import random
            center_pos = [random.randrange(-5, 5), random.randrange(-5, 5), random.randrange(-5, 5)]
            visualize_graph(graph=hierarchy_graph, center_pos=center_pos)
            print(len(graph.hierarchy_graph_map[0]))
            print(len(graph.hierarchy_graph_map[1]))
            print(len(graph.hierarchy_graph_map[2]))
        rate(1)