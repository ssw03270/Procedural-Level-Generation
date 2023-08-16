# gt, labeled, unlabeled
import vpython.no_notebook
from vpython import *
import vpython as vp
import numpy as np

def visualize_primitive_graph(primitive_graph, center_pos):
    node_count = len(primitive_graph.node_list)
    edge_count = len(primitive_graph.adj_map[np.where(primitive_graph.adj_map == 1)]) // 2

    node = [sphere(radius=0.02, color=color.blue) for _ in range(node_count)]
    edge = [cylinder(radius=0.01, color=color.red) for _ in range(0, edge_count)]

    for i, primitive_graph_node in enumerate(primitive_graph.node_list):
        center_node_pos = primitive_graph.node_list[0].node_pos * 1.5 + center_pos
        node[i].pos = vector(*primitive_graph_node.node_pos + center_node_pos)

    edge_index = 0
    for i, primitive_graph_node in enumerate(primitive_graph.node_list):
        center_node_pos = primitive_graph.node_list[0].node_pos * 1.5 + center_pos

        adj_list = primitive_graph.adj_map[primitive_graph_node.node_index]
        adj_list = adj_list[:primitive_graph_node.node_index + 1]
        neighbor_node_list = primitive_graph.node_list[np.where(adj_list == 1)]

        for neighbor_node in neighbor_node_list:
            edge[edge_index].pos = vector(*primitive_graph_node.node_pos + center_node_pos)
            edge[edge_index].axis = vector(*neighbor_node.node_pos + center_node_pos) - vector(
                *primitive_graph_node.node_pos + center_node_pos)

            edge_index += 1

def visualize(graph):
    window = canvas(x=0, y=0, width=1200, height=1200, center=vector(0, 0, 0), background=vector(0.85, 0.85, 0.85))

    while True:
        for primitive_graph in graph.primitive_graph_list:
            visualize_primitive_graph(primitive_graph=primitive_graph, center_pos=[3, 3, 3])

        rate(10)