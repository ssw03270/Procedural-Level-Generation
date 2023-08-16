import numpy as np

from utils.graph_utils import get_adj_map_with_center_node

class Graph():
    def __init__(self, node_list, adj_map, is_primitive=True):
        self.node_list = node_list
        self.adj_map = adj_map
        if not is_primitive:
            self.primitive_graph_list = self.make_primitive_graph()

    def make_primitive_graph(self):
        primitive_graph_list = []

        for cur_node in self.node_list:
            neighbor_node_index_list = np.where(self.adj_map[cur_node.node_index] == 1)[0]
            neighbor_node_list = self.node_list[neighbor_node_index_list]

            cur_node_copy = Node(cur_node.node_pos, 0)
            primitive_node_list = [cur_node_copy]
            for neighbor_node in neighbor_node_list:
                cur_node_pos = cur_node_copy.node_pos
                neighbor_node_pos = neighbor_node.node_pos
                primitive_node_pos = (cur_node_pos + neighbor_node_pos) / 2

                primitive_node = Node(pos=primitive_node_pos, i=len(primitive_node_list))
                primitive_node_list.append(primitive_node)
            primitive_node_list = np.array(primitive_node_list)

            primitive_adj_map = get_adj_map_with_center_node(node_list=primitive_node_list, center_index=0)
            primitive_graph = Graph(node_list=primitive_node_list, adj_map=primitive_adj_map)
            primitive_graph_list.append(primitive_graph)

        return np.array(primitive_graph_list)

class Node():
    def __init__(self, pos, i):
        self.node_pos = np.array(pos)
        self.node_index = i