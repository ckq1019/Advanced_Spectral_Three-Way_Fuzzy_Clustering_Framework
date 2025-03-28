import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def Spectral_Clustering(n_clusters, n_neighborhood=5, Point_matrix=None, Similar_matrix=None, CVI=False):
    '''
    谱聚类
    :param n_clusters:聚类数
    :param n_neighborhood:计算邻接个数
    :param Similar_matrix:相似度矩阵
    :param Point_matrix:点矩阵
    :param CVI:评估指标
    :return:
    '''
    if Similar_matrix is None and Point_matrix is not None:
        Similar_matrix=np.zeros((Point_matrix.shape[0], Point_matrix.shape[0]))
        for i in range(Point_matrix.shape[0]):
            for j in range(i, Point_matrix.shape[0]):
                Similar_matrix[i][j] = Similar_matrix[j][i] = np.linalg.norm(Point_matrix[i] - Point_matrix[j], 2)

    # 利用KNN计算邻接矩阵A
    def AKNN(S, k=10, sigma=1.0):
        N = len(S)
        A = np.zeros((N, N))
        for i in range(N):
            if k == 0:
                neighbours_id = range(i, N)
            else:
                dist_with_index = zip(S[i], range(N))
                # 矩阵的第i行各个元素，与长度为数字N的tuple（元组）集合，组成列表
                dist_with_index = sorted(dist_with_index, key=lambda x: x[0])
                # 按列表的第一个元素升序排列
                neighbours_id = [dist_with_index[m][1] for m in range(k + 1)]
                # xi's k-nearest neighbours
                # 返回K邻近的列坐标
            for j in neighbours_id:  # xj is xi's neighbour
                A[i][j] = np.exp(-S[i][j] ** 2 / 2 / sigma / sigma)
                # 高斯核函数:随着两个向量的距离增大，高斯核函数单调递减
                # np.exp=e^x
                A[j][i] = A[i][j]  # mutually

        return A
    Adjacent = AKNN(Similar_matrix, k=n_neighborhood, sigma=1.3)

    # 计算标准化的拉普拉斯矩阵L,D^(-1/2) L D^(-1/2)
    def calLaplacianMatrix(adjacentMatrix):
        # 计算标准化的拉普拉斯矩阵
        # compute the Degree Matrix: D=sum(A)
        degreeMatrix = np.sum(adjacentMatrix, axis=1)

        # compute the Laplacian Matrix: L=D-A
        laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix
        # np.diag:以一维数组的形式返回方阵的对角线

        # normailze
        # D^(-1/2) L D^(-1/2)
        sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** 0.5))
        # **:返回x的y次幂
        return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)
        # np.dot:矩阵乘法
        # 假设A,B为矩阵，A.dot(B)等价于 np.dot(A,B)

    Laplacian = calLaplacianMatrix(Adjacent)

    # 特征值分解，计算方形矩阵Laplacian的特征值和特征向量, 得到特征向量Hnn,

    x, V = np.linalg.eig(Laplacian)  # x多个特征值组成的一个矢量,V多个特征向量组成的一个矩阵,每一个特征向量都被归一化了,第i列的特征向量v[:,i]对应第i个特征值x[i]
    # x.sort()
    # plt.figure()
    # plt.plot(range(1, len(x)+1), x, "*")
    # plt.show()
    x = zip(x, range(len(x)))
    x = sorted(x, key=lambda x: x[0])
    # 按列表的第一个元素(特征值)升序排列
    H = np.vstack([V[:, i] for (v, i) in x[:n_clusters]]).T  # 取最小的K个特征值对应的特征行向量,按垂直方向（行顺序）堆叠数组构成一个新的数组,然后转置
    # from pyclust import KMedoids
    # kmed = KMedoids(n_clusters=K)
    # center, labels = kmed.fit_predict(H)
    from sklearn.cluster import KMeans, BisectingKMeans, AgglomerativeClustering, Birch, AffinityPropagation, DBSCAN
    # labels = KMeans(n_clusters=n_clusters, init="k-means++").fit_predict(H)
    labels = DBSCAN(eps=0.5, min_samples=30).fit_predict(H)
    # labels = BisectingKMeans(n_clusters=K, init="k-means++").fit_predict(H)
    # labels = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete').fit_predict(H)  # 'ward', 'complete', 'average', 'single'
    # labels = AffinityPropagation().fit_predict(H)
    return labels
    # from FCM import FCM, TCM
    # membership_mat = TCM(H, n_clusters, labels)  # 返回隶属度矩阵
    # # 输出结果
    # result_dict = {-1: []}
    # cluster_core = {}
    # for i in range(membership_mat.shape[0]):
    #     if len(np.where(membership_mat[i] > 0.5)[0]) == 0:
    #         result_dict[-1].append(i)
    #     else:
    #         for j in np.where(membership_mat[i] > 0.5)[0]:
    #             if j not in result_dict.keys():
    #                 result_dict[j] = []
    #                 cluster_core[j] = []
    #             result_dict[j].append(i)
    #             if membership_mat[i][j] > 0.7:
    #                 cluster_core[j].append(i)
    # if CVI:
    #     from Parameter_Selection_methodology import VCIM
    #     cvi = VCIM(result_cluster=result_dict, point_matrix=Point_matrix, membership_mat=membership_mat)
    #     return result_dict, cluster_core, cvi
    # result_cluster = {}
    # for i in range(K):
    #     # result_cluster[int(np.where(H[:, 0] == center[i, 0])[0])] = list(np.where(labels == i)[0])
    #     result_cluster[i] = list(np.where(labels == i)[0])
    # return result_cluster
    # return result_dict, cluster_core


def mySpectral_Clustering(n_clusters, Similar_matrix, Weight):
    '''
    三向谱聚类
    :param n_clusters:聚类数
    :param Similar_matrix:相似度矩阵
    :param Weight:权重矩阵
    :return:
    '''
    if Similar_matrix.shape[-1] == 3:
        Similar_matrix = (Similar_matrix[:, :, 0] ** 2 + Similar_matrix[:, :, 1] ** 2 +
                          Similar_matrix[:, :, 2] ** 2) ** (1/2)

    # W = np.zeros((Similar_matrix.shape[0], Similar_matrix.shape[0]), dtype=np.float64)
    # for i in range(Similar_matrix.shape[0]):
    #     for j in range(i, Similar_matrix.shape[0]):
    #         W[j][i] = W[i][j] = Weight[i] * Weight[j]

    # 计算邻接矩阵A
    def AKNN(S, sigma=1.0):
        N = S.shape[0]
        A = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                A[i][j] = np.exp(-S[i][j] ** 2 / 2 / sigma / sigma)
                # 高斯核函数:随着两个向量的距离增大，高斯核函数单调递减
                # np.exp=e^x
                A[j][i] = A[i][j]  # mutually
        return A

    Adjacent = AKNN(Similar_matrix, sigma=1.3)

    # 计算标准化的拉普拉斯矩阵L,D^(-1/2) L D^(-1/2)
    def calLaplacianMatrix(adjacentMatrix):
        # 计算标准化的拉普拉斯矩阵
        # compute the Degree Matrix: D=sum(A)
        degreeMatrix = np.sum(adjacentMatrix, axis=1)

        # compute the Laplacian Matrix: L=D-A
        laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix
        # np.diag:以一维数组的形式返回方阵的对角线

        # normailze
        # D^(-1/2) L D^(-1/2)
        sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** 0.5))
        # **:返回x的y次幂
        return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)
        # np.dot:矩阵乘法
        # 假设A,B为矩阵，A.dot(B)等价于 np.dot(A,B)

    Laplacian = calLaplacianMatrix(Adjacent)

    # 特征值分解，计算方形矩阵Laplacian的特征值和特征向量, 得到特征向量Hnn,

    x, V = np.linalg.eig(Laplacian)  # x多个特征值组成的一个矢量,V多个特征向量组成的一个矩阵,每一个特征向量都被归一化了,第i列的特征向量v[:,i]对应第i个特征值x[i]
    # x.sort()
    # plt.figure()
    # plt.plot(range(1, len(x)+1), x, "*")
    # plt.show()
    x = zip(x, range(len(x)))
    x = sorted(x, key=lambda x: x[0])
    # 按列表的第一个元素(特征值)升序排列
    H = np.vstack([V[:, i] for (v, i) in x[:n_clusters]]).T  # 取最小的K个特征值对应的特征行向量,按垂直方向（行顺序）堆叠数组构成一个新的数组,然后转置

    from Cluster.FCM import TCM
    return TCM(H, n_clusters, Weight, Membershipmat=True)  # 返回隶属度矩阵


if __name__ == '__main__':
    file_path = r"cyclonic_[353  80].csv"
    from main import read_data
    origin_data, file_format = read_data(os.path.join(r'D:\Trajectory_analysis\Data', file_path))
    from Data_Preprocessing import format_conversion
    data = format_conversion(origin_data, file_format='CSV', Data_processing=False)
    track_list = []
    for i in data.keys():
        track_list.append(data[i])

    from Similarity.Similarity_Measurement_Methodology import SimilarityMatrix
    sm = SimilarityMatrix(isCalculate=False, TR_List=track_list, K=0, reshuffle=False,
                          filepath=os.path.join(r'D:\Trajectory_analysis\Similarity\output', 'Similarity_matrix.npy'))
    similar_matrix = sm.similarity_matrix

    # membership_mat = np.load(os.path.join(r"D:\Trajectory_analysis\Intensive_Time_Extraction\output",
    #                                       "Intensive_time_interval_membership.npy"))

    # 降噪
    # k_distance = []
    # for i in range(similar_matrix.shape[0]):
    #     k_distance.append(similar_matrix[i, np.argsort(similar_matrix[i, :, 0])[6], 0])
    # k_distance.sort()
    # plt.figure()
    # plt.plot(np.arange(len(k_distance)), k_distance)
    # plt.show()
    # plt.close()
    # from Dimension_Reduction.Dimension_Reduction_Methodology import SimilarMatrixProjection
    # point_matrix = SimilarMatrixProjection(algorithm='MDS', n_components=1, reshuffle=False, filepath=
    # os.path.join(r'D:\Trajectory_analysis\Dimension_Reduction\output', 'all_point_matrix.npy')).fit(similar_matrix)
    #
    # for i in range(similar_matrix.shape[0]):
    #     for j in range(similar_matrix.shape[0]):
    #         similar_matrix[i, j, 0] = math.sqrt(similar_matrix[i, j, 0] ** 2 + similar_matrix[i, j, 1] ** 2 +
    #                                             similar_matrix[i, j, 2] ** 2)
    #
    # from sklearn.neighbors import LocalOutlierFactor
    # label = LocalOutlierFactor(n_neighbors=5).fit_predict(point_matrix)
    # result = []
    # similar_matrix = similar_matrix[label != -1, :, :]
    # similar_matrix = similar_matrix[:, label != -1, :]
    # point_matrix = point_matrix[label != -1]
    # from sklearn.manifold import MDS
    # point_matrix = MDS(n_components=2, dissimilarity="precomputed").fit_transform(similar_matrix)
    W = []
    for i in range(len(track_list)):
        W.append(track_list[i].time_membership)
    W = np.array(W)
    sc = []

    def SC(memership_mat, similar_mat):
        s = []
        for cluster_id in range(memership_mat.shape[1]):  # 遍历每个簇
            for track_id in np.where(memership_mat[:, cluster_id] == 1)[0]:  # 遍历每个簇中的每个点
                # 保存点到簇内距离的平均值
                a = similar_mat[track_id, np.where(memership_mat[:, cluster_id] == 1)[0]]
                a = a[a != 0]  # 去掉自身
                # 保存每个点到其他簇的平均距离
                b = np.mean(similar_mat[track_id, np.where((memership_mat[:, cluster_id] == 0) &
                                                           (np.sum(memership_mat, axis=1) != 0))[0]])
                s.append((np.min(b) - np.mean(a)) / max(np.mean(a), np.min(b)))
        return np.mean(s)

    a1 = 0
    wrong = False
    for i in range(2, round(math.sqrt(similar_matrix.shape[0]))):
        result = mySpectral_Clustering(n_clusters=i, Similar_matrix=similar_matrix[:, :, a1], Weight=W)
        for col in range(result.shape[1]):
            if len(np.where(result[:, col] == 1)[0]) < 2:
                print("{} break".format(i))
                wrong = True
                break
        if wrong:
            break
        similar_mat = similar_matrix[np.where(result == 1)[0]]
        similar_mat = similar_mat[:, np.where(result == 1)[0]]
        similar_mat = similar_mat[:, :, a1]
        sc.append(SC(result[np.where(result == 1)[0]], similar_mat))
        print(sc)
    plt.figure()
    plt.plot(np.arange(2, len(sc)+2), sc, "ro-")
    plt.show()
        # label[label != -1] = Spectral_Clustering(n_clusters=i, n_neighborhood=5,
        #                                          Point_matrix=point_matrix[label != -1],
        #                                          Similar_matrix=None, CVI=False)
        # from sklearn.cluster import SpectralClustering
        # label[label != -1] = SpectralClustering(n_clusters=i, gamma=1.3, affinity="precomputed", n_neighbors=5).\
        #     fit(similar_matrix[i, j, 0]).labels_
        # from Cluster.Trajectory_Clustering_Methodology import consolidation
        # result_cluster = consolidation(result_dict)
    #     from Cluster.Parameter_Selection_methodology import Evaluation_Model
    #     if len(result_dict[0]) == 0:
    #         break
    #     print("i:{}".format(i))
    #     parameter.append([i] + Evaluation_Model(result_cluster=result_dict, similar_matrix=similar_matrix,
    #                                             point_matrix=None, label_=None))
    # #     # parameter.append([i] + [cvi])
    #     print('k:', i, "\nresult_cluster:", result_dict)
    #     from main import result_integration
    #     result_integration(all_result, result_dict, track_list)  # 把聚类结果进行整合
    #     print(all_result)
    #     from Data_Analysis.Data_Analysis import Similarity_analysis
    #     result.append(Similarity_analysis(data, all_result, Draw=False))
    #     print("-" * 20)
    # from Cluster.Trajectory_Clustering_Methodology import Parameter_Selection_Line
    # Parameter_Selection_Line(parameter, file=r"D:\Trajectory_analysis\Cluster\output\Parameter_Selection.html")
    # result = np.array(result, dtype=np.float64)
    # print(np.max(result, axis=0))
