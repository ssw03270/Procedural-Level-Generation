# level 정보는 인접행렬과 level 을 구성하는 object 가 node 로 표현되어 list 로 존재함
# node 의 feature 에는 (obj_idx, pos_x, pos_y, pos_z, scale_x, scale_y, scale_z)
# rule 정보는 node 하나와 이 node 주변의 인접한 object 에 해당하는 node 로 구성
# rule 의 경우, root node 의 중심 위치는 0, 0, 0
# level 정보를 하나의 단일 된 feature 로 보내야함
# 그 이전에 level 정보를 message passing 할 것임
# 이는 aggregate 과정과 update 과정으로 구성됨
# aggregate 를 통해 인접 node 들의 정보를 각 중심 node 로 보내줄 것이고 (학습 들어감)
# 중심 node 와 aggregate 된 중심 노드를 바탕으로 중심 node 를 업데이트 함 (학습 들어감)
# 마지막으로 update 된 level 을 readout function 을 통해 하나로 더해서 표현해야 함 (학습 들어감)
# unity 를 통해 rule 을 추출할 때, rules 의 adj_matrix 와 node_feature_list 를 만들어야 함