import numpy as np

def get_adj_map_with_dist(node_list, dist):
    node_count = len(node_list)
    adj_map = np.zeros((node_count, node_count))
    for cur_node in node_list:
        for tar_node in node_list:
            if np.linalg.norm(cur_node.node_pos - tar_node.node_pos) == dist:
                adj_map[cur_node.node_index, tar_node.node_index] = 1

    return adj_map

def get_adj_map_with_center_node(node_list, center_index):
    node_count = len(node_list)
    adj_map = np.zeros((node_count, node_count))
    for i in range(node_count):
        if center_index != i:
            adj_map[center_index, i] = 1
            adj_map[i, center_index] = 1

    return adj_map

def get_adj_map_with_edge_list(node_list, edge_list):
    node_count = len(node_list)
    adj_map = np.zeros((node_count, node_count))
    for edge in edge_list:
        adj_map[edge[0], edge[1]] = 1
        adj_map[edge[1], edge[0]] = 1

    return adj_map

def get_edge_list_with_adj_map(adj_map):
    edge_list = []
    for i, adj_list in enumerate(adj_map):
        for j in np.where(adj_list == 1)[0]:
            edge = [i, j]

            if [j, i] not in edge_list:
                edge_list.append(edge)

    edge_list = np.array(edge_list)
    return edge_list