import numpy as np

from utils.graph_utils import get_adj_map_with_center_node, get_edge_list_with_adj_map, get_adj_map_with_edge_list

class Graph():
    def __init__(self, node_list, adj_map, is_primitive=True):
        self.node_list = node_list
        self.adj_map = adj_map
        if not is_primitive:
            self.primitive_graph_list = self.make_primitive_graph()
            self.hierarchy_graph_map = self.make_hierarchy_graph(max_iter=3)

    def make_primitive_graph(self):
        primitive_graph_list = []

        for cur_node in self.node_list:
            neighbor_node_index_list = np.where(self.adj_map[cur_node.node_index] == 1)[0]
            neighbor_node_list = self.node_list[neighbor_node_index_list]

            cur_node_copy = Node(pos=cur_node.node_pos, idx=0, is_center=True)

            # make primitive node
            primitive_node_list = [cur_node_copy]
            for neighbor_node in neighbor_node_list:
                cur_node_pos = cur_node_copy.node_pos
                neighbor_node_pos = neighbor_node.node_pos
                primitive_node_pos = (cur_node_pos + neighbor_node_pos) / 2

                primitive_node = Node(pos=primitive_node_pos, idx=len(primitive_node_list))
                primitive_node_list.append(primitive_node)
            primitive_node_list = np.array(primitive_node_list)

            # make primitive graph
            primitive_adj_map = get_adj_map_with_center_node(node_list=primitive_node_list, center_index=0)
            primitive_graph = Graph(node_list=primitive_node_list, adj_map=primitive_adj_map)
            primitive_graph_list.append(primitive_graph)

        return np.array(primitive_graph_list)

    def make_hierarchy_graph(self, max_iter):
        hierarchy_graph_map = []
        for iter in range(max_iter):
            hierarchy_graph_list = []

            if iter == 0:
                hierarchy_graph_list = self.primitive_graph_list
                hierarchy_graph_map.append(hierarchy_graph_list)
                continue

            for left_hierarchy_graph in hierarchy_graph_map[iter - 1]:
                for right_primitive_graph in self.primitive_graph_list:
                    for left_hierarchy_node in left_hierarchy_graph.node_list:
                        for right_primitive_node in right_primitive_graph.node_list:
                            # cant choose center node
                            if not left_hierarchy_node.is_center and not right_primitive_node.is_center:
                                right_primitive_edge_list = get_edge_list_with_adj_map(adj_map=right_primitive_graph.adj_map) + len(left_hierarchy_graph.node_list)
                                current_right_primitive_node_index = right_primitive_node.node_index + len(left_hierarchy_graph.node_list)
                                current_left_hierarchy_node_index = left_hierarchy_node.node_index

                                # make new right primitive edge list
                                new_right_primitive_edge_list = []
                                for right_primitive_edge in right_primitive_edge_list:
                                    new_edge = right_primitive_edge
                                    if new_edge[0] == current_right_primitive_node_index:
                                        new_edge[0] = current_left_hierarchy_node_index
                                    elif new_edge[1] == current_right_primitive_node_index:
                                        right_primitive_edge[1] = current_left_hierarchy_node_index

                                    if new_edge[0] > current_right_primitive_node_index:
                                        new_edge[0] -= 1
                                    if new_edge[1] > current_right_primitive_node_index:
                                        new_edge[1] -= 1

                                    new_right_primitive_edge_list.append(new_edge)

                                # if already exist
                                is_exist = False

                                # make new node list
                                new_node_list = left_hierarchy_graph.node_list.tolist()
                                for right_primitive_node2 in right_primitive_graph.node_list:
                                    new_idx = right_primitive_node2.node_index + len(left_hierarchy_graph.node_list)
                                    if new_idx != current_right_primitive_node_index:
                                        if new_idx == current_right_primitive_node_index:
                                            continue

                                        if new_idx > current_right_primitive_node_index:
                                            new_idx -= 1

                                        pos_diff = left_hierarchy_node.node_pos - right_primitive_node.node_pos
                                        new_pos = right_primitive_node2.node_pos + pos_diff
                                        right_primitive_node_copy = Node(pos=new_pos,
                                                                         idx=new_idx, is_center=False)

                                        for node in new_node_list:
                                            if node.node_pos.tolist() == new_pos.tolist():
                                                is_exist = True

                                        new_node_list.append(right_primitive_node_copy)
                                new_node_list = np.array(new_node_list)

                                if is_exist:
                                    continue

                                # make new adj map
                                right_primitive_adj_map = get_adj_map_with_edge_list(node_list=new_node_list, edge_list=new_right_primitive_edge_list)
                                new_adj_map = right_primitive_adj_map
                                new_adj_map[:len(left_hierarchy_graph.node_list), :len(left_hierarchy_graph.node_list)] = left_hierarchy_graph.adj_map

                                # make hierarchy graph
                                hierarchy_graph = Graph(node_list=new_node_list, adj_map=new_adj_map)
                                hierarchy_graph_list.append(hierarchy_graph)

            hierarchy_graph_map.append(hierarchy_graph_list)

        return hierarchy_graph_map


class Node():
    def __init__(self, pos, idx, is_center=False):
        self.node_pos = np.array(pos)
        self.node_index = idx
        self.is_center = is_center