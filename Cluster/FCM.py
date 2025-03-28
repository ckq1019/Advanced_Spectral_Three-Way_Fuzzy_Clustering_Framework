import numpy as np


def FCM(X, c_clusters, m=2, eps=0.0001, cv=False, Membership=False, cluster_speilon=0.6):
    '''
    :param X: 数据集
    :param c_clusters: 聚类数目
    :param m:
    :param eps: 属于当隶属矩阵小于某个值的时候停止迭代
    :param cv: 变异系数
    :param Membership:隶属度矩阵，uij是xi在类j的隶属度
    :param cluster_speilon:隶属度大于某值时候就属于某个聚类
    :return:
    '''
    membership_mat = np.random.random((len(X), c_clusters))
    membership_mat = np.divide(membership_mat, np.sum(membership_mat, axis=1)[:, np.newaxis])

    while True:
        # 计算簇中心
        working_membership_mat = membership_mat ** m
        Centroids = np.divide(np.dot(working_membership_mat.T, X),
                              np.sum(working_membership_mat.T, axis=1)[:, np.newaxis])

        n_c_distance_mat = np.zeros((len(X), c_clusters))
        for i, x in enumerate(X):
            for j, c in enumerate(Centroids):
                n_c_distance_mat[i][j] = np.linalg.norm(x - c, 2)

        new_membership_mat = np.zeros((len(X), c_clusters))

        for i, x in enumerate(X):
            for j, c in enumerate(Centroids):
                # 通过之前的簇中心，更新u
                new_membership_mat[i][j] = 1. / np.sum((n_c_distance_mat[i][j] / n_c_distance_mat[i]) ** (2 / (m - 1)))

        if np.sum(abs(new_membership_mat - membership_mat)) < eps:
            break
        membership_mat = new_membership_mat
    if cv is True:
        return np.std(np.dot(new_membership_mat.T, n_c_distance_mat)) / np.mean(np.dot(new_membership_mat.T,
                                                                                       n_c_distance_mat))

    if Membership is True:
        return new_membership_mat
    result = {}
    for i in range(0, c_clusters):
        result[i] = list(np.where(new_membership_mat[:, i] > cluster_speilon)[0])
    return result
    # label = np.zeros((len(X)))
    # for i in range(1, c_clusters+1):
    #     label[np.where(new_membership_mat[:, i-1] > cluster_speilon)[0]] = i
    # return label


def TCM(X, n_clusters, Weight, Membershipmat=False):
    '''
    :param X: 数据集
    :param n_clusters: 聚类数目
    :param Weight:该轨迹隶属该簇的隶属度
    :param Membershipmat:如果为True的话就返回隶属读矩阵
    :return:
    '''
    membership_mat = np.random.random((len(X), n_clusters))  # 初始化随机隶属度
    membership_mat = np.divide(membership_mat, np.sum(membership_mat, axis=1)[:, np.newaxis])  # 归一化
    Weight = np.repeat(Weight, n_clusters).reshape((-1, n_clusters))
    # G = membership_mat / np.max(membership_mat, axis=0)  # 聚类相对隶属度
    # membership_mat[G > 0.7] = 1
    # membership_mat[
    #     np.where(membership_mat == 1)[0], np.where(membership_mat[np.where(membership_mat == 1)[0]] != 1)[1]] = 0
    # membership_mat[G < 0.3] = 0
    # membership_mat = np.divide(membership_mat, np.sum(membership_mat, axis=1)[:, np.newaxis])
    membership_mat = Weight * membership_mat
    while True:
        # 计算簇中心
        working_membership_mat = membership_mat ** 2
        Centroids = np.divide(np.dot(working_membership_mat.T, X),
                              np.sum(working_membership_mat.T, axis=1)[:, np.newaxis])

        n_c_distance_mat = np.zeros((len(X), n_clusters))
        for i, x in enumerate(X):
            for j, c in enumerate(Centroids):
                n_c_distance_mat[i][j] = np.linalg.norm(x - c, 2)

        new_membership_mat = np.zeros((len(X), n_clusters))

        for i, x in enumerate(X):
            for j, c in enumerate(Centroids):
                # 通过之前的簇中心，更新u
                new_membership_mat[i][j] = 1. / np.sum((n_c_distance_mat[i][j] / n_c_distance_mat[i]) ** 2)

        new_membership_mat = Weight * new_membership_mat
        for i in range(new_membership_mat.shape[0]):
            epsilon = 0
            for j in np.argsort(-new_membership_mat[i]):
                if epsilon >= 0.8 * Weight[i][0] or new_membership_mat[i][j] < max(new_membership_mat[i])/4:
                    new_membership_mat[i][j] = 0
                epsilon += new_membership_mat[i][j]
            # if np.mean(new_membership_mat[i]) < 0.1 or Weight[i][0] < 0.2:
            #     new_membership_mat[i] = 0
            #     continue
            # 数据归一化
            if sum(new_membership_mat[i]) != 0:
                new_membership_mat[i] = new_membership_mat[i] / sum(new_membership_mat[i]) * Weight[i]

        # G = new_membership_mat / np.max(new_membership_mat, axis=0)  # 聚类相对隶属度
        # new_membership_mat[G > 0.7] = 1
        # new_membership_mat[np.where(new_membership_mat == 1)[0],
        #                    np.where(new_membership_mat[np.where(new_membership_mat == 1)[0]] != 1)[1]] = 0
        # new_membership_mat[G < 0.3] = 0
        # new_membership_mat[np.where(np.sum(new_membership_mat, axis=1) != 0)[0]] = \
        #     np.divide(new_membership_mat[np.where(np.sum(new_membership_mat, axis=1) != 0)[0]],
        #               np.sum(new_membership_mat[np.where(np.sum(new_membership_mat, axis=1) != 0)[0]], axis=1)
        #               [:, np.newaxis])
        # new_membership_mat = Weight * new_membership_mat
        if np.sum(abs(membership_mat - new_membership_mat)) < 0.001:
            break
        membership_mat = new_membership_mat

    if Membershipmat:
        return membership_mat
    else:  # 返回聚类结果
        result = {}
        for i in range(0, n_clusters):
            result[i] = list(np.where(new_membership_mat[:, i] == 1)[0])
        return result


# def TCM(X, n_cluster, labels):
#     '''
#     :param X: 数据集
#     :param n_cluster: 聚类数目
#     :param labels:硬聚类结果
#     :return: 返回隶属度矩阵
#     '''
#     # 1. 初始化簇心
#     CENTER = np.zeros((n_cluster, X.shape[1]))
#     for cluster_id in range(n_cluster):
#         CENTER[cluster_id] = np.mean(X[labels == cluster_id], axis=0)
#
#     # 2. 根据初始化的簇心，计算（收敛后的）隶属度
#     membership_mat = np.zeros((len(X), n_cluster))  # 隶属度矩阵
#     distance = np.zeros((len(X), n_cluster))  # 计算距离簇心的距离（簇中）
#     for i, x in enumerate(X):
#         for j, c in enumerate(CENTER):
#             if labels[i] == j:
#                 distance[i][j] = np.linalg.norm(x - c, 2)
#     # 根据距离计算隶属度
#     for i, x in enumerate(X):
#         for j, c in enumerate(CENTER):
#             if labels[i] == j:
#                 membership_mat[i][j] = np.max(distance[distance[:, j] != 0, j]) / distance[i][j]
#
#     membership_mat = membership_mat / np.sum(membership_mat, axis=1)
#     # 根据计算得到的隶属度计算平均值
#     working_membership_mat = membership_mat ** 2
#     CENTER = np.dot(working_membership_mat.T, X)  # 更新簇心
#
#     # 根据计算得到的新的簇心，来计算整个矩阵的隶属度
#     n_c_distance_mat = np.zeros((len(X), n_cluster))
#     for i, x in enumerate(X):
#         for j, c in enumerate(CENTER):
#             n_c_distance_mat[i][j] = np.linalg.norm(x - c, 2)
#
#     new_membership_mat = np.zeros((len(X), n_cluster))
#
#     for i, x in enumerate(X):
#         for j, c in enumerate(CENTER):
#             # 通过之前的簇中心，更新u
#             new_membership_mat[i][j] = 1. / np.sum((n_c_distance_mat[i][j] / n_c_distance_mat[i]) ** 2)
#
#     # return new_membership_mat
#     result = {-1: []}
#     core = {}
#     for i in range(len(X)):
#         if len(np.where(new_membership_mat[i] > 0.5)[0]) == 0:
#             result[-1].append(i)
#         for j in np.where(new_membership_mat[i] > 0.5)[0]:
#             if j not in result.keys():
#                 result[j] = []
#                 core[j] = []
#             result[j].append(i)
#             if new_membership_mat[i][j] > 0.7:
#                 core[j].append(i)
#     return result, core

if __name__ == '__main__':
    print(1)