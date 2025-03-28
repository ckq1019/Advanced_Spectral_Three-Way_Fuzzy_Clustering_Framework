import numpy as np
from Dimension_Reduction.MDS import MDS


class ISOMAP:
    # 等度量映射（Isometric Mapping，简称Isomap）
    def __init__(self, n_components):
        self.n_components = n_components

    @staticmethod
    def Dijkstra(similar_matrix):
        # 根据相似度得到最小联通图
        # 调用Dijkstra算法最短路径算法计算任意两个样本之间的距离dist(xi,xj)，获得距离矩阵
        n = similar_matrix.shape[0]

        def adjacency_ditance(point_index):
            # 返回point_index的邻接距离[value, index]
            result = {}  # keys:(a_id,b_id) , value: diatance
            adjacency_index = []
            for i in np.argsort(similar_matrix[point_index]):
                if i in np.nonzero(similar_matrix[point_index])[0]:
                    if np.isnan(similar_matrix[point_index][i]):
                        break
                    result[(point_index, i)] = similar_matrix[point_index][i]
                    adjacency_index.append(i)
            return result, adjacency_index

        def update_ajdiatance(distance, aj_dis, point_index, adjacency_index):
            # 更新distance字典
            delete_keys = []
            for i_index in aj_dis.keys():
                if i_index[1] in point_index:  # 如果添加的该数据已经diatance
                    if aj_dis[i_index] < distance[list(distance.keys())[point_index.index(i_index[1])]]:
                        distance[i_index] = aj_dis[i_index]
                        delete_keys.append(list(distance.keys())[point_index.index(i_index[1])])
                else:
                    distance[i_index] = aj_dis[i_index]
            for i in delete_keys:
                distance.pop(i)
            # 对distance进行排序
            return dict(sorted(distance.items(), key=lambda x: x[1]))

        def update_ajpoint(distance):
            # 更新邻接aj_index
            result = []
            for i in distance.keys():
                result.append(i[1])
            return result

        all = []
        result = []
        for i in range(n):
            if i in all:
                continue
            queue = [i]  # 存放当前已访问的点的队列
            distance, ajp = adjacency_ditance(i)  # 存放visited队列中所有邻接未访问点的距离[VALUE,INDEX]
            while 1:
                saj_index = list(distance.keys())[0]
                distance.pop(saj_index)  # 把最小距离asj取出index
                ajp.remove(saj_index[1])
                saj_index = saj_index[1]
                queue.append(saj_index)  # 将最小值index加入队列中,distance中的最小vlue取出

                aj_dis, ajpp = adjacency_ditance(saj_index)  # 把saj_index的邻接距离
                delete_keys = []
                for j_index in aj_dis.keys():
                    if j_index[1] in queue:
                        delete_keys.append(j_index)
                        continue
                for del_keys in delete_keys:
                    aj_dis.pop(del_keys)
                    ajpp.remove(del_keys[1])
                # 刷新distance
                distance = update_ajdiatance(distance, aj_dis, ajp, ajpp)
                if len(distance.keys()) == 0:
                    result.append(queue)
                    all.extend(queue)
                    print("{}连通".format(queue))
                    break
                else:
                    ajp = update_ajpoint(distance)
        return result

    @staticmethod
    def Extract_Corresponding_Value(tr_list, similar_matrix, Minimum_connected):
        # 根据Minimum_connected的id对应的list, 相似度矩阵
        partial_tr_list = []
        for i in Minimum_connected:
            partial_tr_list.append(tr_list[i])
        partial_similar_matrix = similar_matrix[Minimum_connected, :, :].copy()
        partial_similar_matrix = partial_similar_matrix[:, Minimum_connected, :]
        return partial_tr_list, partial_similar_matrix.reshape((len(Minimum_connected), len(Minimum_connected), 3))

    def fit(self, similar_matrix):
        n = similar_matrix.shape[0]
        similar_matrix[range(n), range(n)] = 0  # 设置对角线的值0

        # 调用Dijkstra算法最短路径算法计算任意两个样本之间的距离dist(xi,xj)，获得距离矩阵
        def adjacency_ditance(point_index):
            # 返回point_index的邻接距离[value, index]
            result = {}  # keys:(a_id,b_id) , value: diatance
            adjacency_index = []
            for i in np.argsort(similar_matrix[point_index]):
                if i in np.nonzero(similar_matrix[point_index])[0]:
                    if np.isnan(similar_matrix[point_index][i]):
                        break
                    result[(point_index, i)] = similar_matrix[point_index][i]
                    adjacency_index.append(i)
            return result, adjacency_index

        def update_ajdiatance(distance, aj_dis, point_index, adjacency_index):
            # 更新distance字典
            delete_keys = []
            for i_index in aj_dis.keys():
                if i_index[1] in point_index:  # 如果添加的该数据已经diatance
                    if aj_dis[i_index] < distance[list(distance.keys())[point_index.index(i_index[1])]]:
                        distance[i_index] = aj_dis[i_index]
                        delete_keys.append(list(distance.keys())[point_index.index(i_index[1])])
                else:
                    distance[i_index] = aj_dis[i_index]
            for i in delete_keys:
                distance.pop(i)
            # 对distance进行排序
            return dict(sorted(distance.items(), key=lambda x: x[1]))

        def update_ajpoint(distance):
            # 更新邻接aj_index
            result = []
            for i in distance.keys():
                result.append(i[1])
            return result

        for i in range(n):
            queue = [i]  # 存放当前已访问的点的队列
            distance, ajp = adjacency_ditance(i)  # 存放visited队列中所有邻接未访问点的距离[VALUE,INDEX]
            while len(similar_matrix[np.isnan(similar_matrix[i])]):
                saj_index = list(distance.keys())[0]
                distance.pop(saj_index)  # 把最小距离asj取出index
                ajp.remove(saj_index[1])
                saj_index = saj_index[1]
                queue.append(saj_index)  # 将最小值index加入队列中,distance中的最小vlue取出

                aj_dis, ajpp = adjacency_ditance(saj_index)  # 把saj_index的邻接距离
                delete_keys = []
                for j_index in aj_dis.keys():
                    if j_index[1] in queue:
                        delete_keys.append(j_index)
                        continue
                    if np.isnan(similar_matrix[i, j_index[1]]) or \
                            similar_matrix[i, j_index[0]] + aj_dis[j_index] < similar_matrix[i, j_index[1]]:
                        similar_matrix[i][j_index[1]] = similar_matrix[i][j_index[0]] + aj_dis[j_index]  # 刷新距离矩阵

                for del_keys in delete_keys:
                    aj_dis.pop(del_keys)
                    ajpp.remove(del_keys[1])
                # 刷新distance
                distance = update_ajdiatance(distance, aj_dis, ajp, ajpp)
                if len(distance.keys()) == 0:
                    break
                    print("{} 不连接！".format(queue))
                    return None
                else:
                    ajp = update_ajpoint(distance)

        # 调用MDS算法获得样本集在低维空间中的矩阵Z
        # def mds(dist, n_dims):
        #     # dist (n_samples, n_samples)
        #     dist = dist ** 2
        #     n = dist.shape[0]
        #     T1 = np.ones((n, n)) * np.sum(dist) / n ** 2
        #     T2 = np.sum(dist, axis=1) / n
        #     T3 = np.sum(dist, axis=0) / n
        #
        #     B = -(T1 - T2 - T3 + dist) / 2
        #
        #     eig_val, eig_vector = np.linalg.eig(B)
        #     index_ = np.argsort(-eig_val)[:n_dims]
        #     picked_eig_val = eig_val[index_].real
        #     picked_eig_vector = eig_vector[:, index_]
        #     return picked_eig_vector * picked_eig_val ** (0.5)
        # mds(similar_matrix, self.n_components)
        from sklearn.manifold import MDS
        return MDS(n_components=self.n_components, dissimilarity="precomputed").fit_transform(similar_matrix)
        # return MDS(1).fit(similar_matrix)


def dijkstra(matrix, start_node):
    matrix_length = len(matrix)  # 矩阵一维数组的长度，即节点的个数

    used_node = [False] * matrix_length  # 访问过的节点数组

    distance = [np.inf] * matrix_length  # 最短路径距离数组

    distance[start_node] = 0  # 初始化，将起始节点的最短路径修改成0

    # 将访问节点中未访问的个数作为循环值，其实也可以用个点长度代替。
    while used_node.count(False):
        min_value = np.inf
        min_value_index = -1

        # 在最短路径节点中找到最小值，已经访问过的不在参与循环。
        # 得到最小值下标，每循环一次肯定有一个最小值
        for index in range(matrix_length):
            if not used_node[index] and distance[index] < min_value:
                min_value = distance[index]
                min_value_index = index

        # 将访问节点数组对应的值修改成True，标志其已经访问过了
        used_node[min_value_index] = True

        # 更新distance数组。
        # 以B点为例：distance[x] 起始点达到B点的距离。
        # distance[min_value_index] + matrix[min_value_index][index] 是起始点经过某点达到B点的距离，比较两个值，取较小的那个。
        for index in range(matrix_length):
            distance[index] = min(distance[index], distance[min_value_index] + matrix[min_value_index][index])

    return distance

matrix_ = [
    [0,10,np.inf,4,np.inf,np.inf],
    [10,0,8,2,6,np.inf],
    [np.inf,8,10,15,1,5],
    [4,2,15,0,6,np.inf],
    [np.inf,6,1,6,0,12],
    [np.inf,np.inf,5,np.inf,12,0]
]
ret = dijkstra(matrix_, 0)
print(ret)