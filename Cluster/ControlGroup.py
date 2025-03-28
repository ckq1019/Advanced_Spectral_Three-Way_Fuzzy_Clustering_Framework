import numpy as np
import os


class ConsensusSpectralFuzzyCoClustering:  # 不用三向隶属度的结果模糊SC
    def __init__(self, data_num, n_clusters, weight):
        self.data_num = data_num
        self.n_clusters = n_clusters  # 针对三个特征的分别聚类数
        self.feature_num = 3
        self.weight = weight  # 三向隶属度
        self.feature0 = np.zeros((data_num, n_clusters[0]))  # 特征0针对各个聚类的隶属度
        self.feature1 = np.zeros((data_num, n_clusters[1]))
        self.feature2 = np.zeros((data_num, n_clusters[2]))
        self.estimate = -1  # 对每次聚类结果进行评估，越大越好
        self.feature = [self.feature0, self.feature1, self.feature2]  # 个体特征
        self.cocluster_membership = np.zeros((data_num, n_clusters[0] * n_clusters[1] * n_clusters[2]))
        self.cluster_title = {}  # 共聚类隶属度的表头标签（就是分别代表三个特征的聚类标签）

    def fit(self, Similar_matrix, Weight):  # 聚类
        self.feature0 = self.spectralclustering(self.n_clusters[0], Similar_matrix[:, :, 0], Weight)
        self.feature1 = self.spectralclustering(self.n_clusters[1], Similar_matrix[:, :, 1], Weight)
        self.feature2 = self.spectralclustering(self.n_clusters[2], Similar_matrix[:, :, 2], Weight)

    def spectralclustering(self, n_clusters, similar_matrix, weight):
        w = np.zeros((similar_matrix.shape[0], similar_matrix.shape[0]), dtype=np.float64)
        for i in range(similar_matrix.shape[0]):
            for j in range(i, similar_matrix.shape[0]):
                w[j][i] = w[i][j] = weight[i] * weight[j]

        def AKNN(S, sigma=1.0):
            N = S.shape[0]
            A = np.zeros((N, N))
            for i in range(N):
                for j in range(i, N):
                    A[i][j] = np.exp(-S[i][j] ** 2 / 2 / sigma / sigma) * w[i, j]
                    A[j][i] = A[i][j]
            return A
        Adjacent = AKNN(similar_matrix, sigma=1.3)

        def calLaplacianMatrix(adjacentMatrix):
            # 计算标准化的拉普拉斯矩阵
            degreeMatrix = np.sum(adjacentMatrix, axis=1)
            laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix
            sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** 0.5))
            return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)

        Laplacian = calLaplacianMatrix(Adjacent)

        x, V = np.linalg.eig(Laplacian)

        x = zip(x, range(len(x)))
        x = sorted(x, key=lambda x: x[0])
        # 按列表的第一个元素(特征值)升序排列
        H = np.vstack([V[:, i] for (v, i) in x[:n_clusters]]).T  # 取最小的K个特征值对应的特征行向量,按垂直方向（行顺序）堆叠数组构成一个新的数组,然后转置

        # 模糊聚类
        membership_mat = np.random.random((len(H), n_clusters))  # 初始化随机隶属度
        membership_mat = np.divide(membership_mat, np.sum(membership_mat, axis=1)[:, np.newaxis])  # 归一化
        W = np.repeat(weight, n_clusters).reshape((-1, n_clusters))
        membership_mat = W * membership_mat
        while True:
            # 计算簇中心
            working_membership_mat = membership_mat ** 2
            Centroids = np.divide(np.dot(working_membership_mat.T, H),
                                  np.sum(working_membership_mat.T, axis=1)[:, np.newaxis])

            n_c_distance_mat = np.zeros((len(H), n_clusters))
            for i, x in enumerate(H):
                for j, c in enumerate(Centroids):
                    n_c_distance_mat[i][j] = np.linalg.norm(x - c, 2)

            new_membership_mat = np.zeros((len(H), n_clusters))

            for i, x in enumerate(H):
                for j, c in enumerate(Centroids):
                    # 通过之前的簇中心，更新u
                    new_membership_mat[i][j] = 1. / np.sum((n_c_distance_mat[i][j] / n_c_distance_mat[i]) ** 2)

            new_membership_mat = W * new_membership_mat
            if np.sum(abs(membership_mat - new_membership_mat)) < 0.0001:
                break
            membership_mat = new_membership_mat
        return membership_mat

    def evaluate(self, similar_mat):  # 对这次结果进行评估，并返回评估值
        sc = 0
        sc += self.cluster(0, similar_mat[:, :, 0])  # 分别对三个特征和共聚类结果进行评估
        sc += self.cluster(1, similar_mat[:, :, 1])
        sc += self.cluster(2, similar_mat[:, :, 2])
        sc += self.co_cluster(similar_mat)
        self.estimate = sc / 4
        return sc / 4

    def cluster(self, feature_index, similar_mat):  # 分别根据不同的特征隶属度得到模糊SC，越大越好
        if feature_index == 0:
            feature = self.feature0
        elif feature_index == 1:
            feature = self.feature1
        elif feature_index == 2:
            feature = self.feature2
        s = []
        all_cluster = list(range(self.n_clusters[feature_index]))  # 所有簇
        for cluster_id in all_cluster:  # 遍历每个簇
            inner_membership_mat = feature[:, cluster_id].reshape((-1, 1)) * feature[:, cluster_id].reshape((1, -1))
            for track_id in np.where(feature[:, cluster_id] != 0)[0]:  # 遍历每个簇中的每个数据
                # 保存点到簇内距离的平均值
                dis = inner_membership_mat[track_id] * similar_mat[track_id]
                a = dis[dis != 0]  # 簇内距离
                a = np.sum(a) / np.sum(inner_membership_mat[track_id, dis != 0].flatten())  # 模糊平均簇内距离
                b = []  # 存放其他簇的平均模糊距离
                for other_cluster in all_cluster:
                    if other_cluster == cluster_id:
                        continue
                    out_membership_mat = feature[:, other_cluster] * feature[track_id, cluster_id]
                    dis = out_membership_mat * similar_mat[track_id]  # 簇外距离
                    b.append(np.sum(dis[dis != 0]) / np.sum(out_membership_mat[dis != 0].flatten()))
                s.append((np.min(b) - a) / max(a, np.min(b)))
        print("SC of feature{}:".format(feature_index), np.nanmean(s))
        return np.nanmean(s)

    def co_cluster(self, similar_mat):  # 共聚类评估结果模糊SC越大越好
        # 计算共聚类隶属度
        title = 0
        for a in range(self.n_clusters[0]):
            for b in range(self.n_clusters[1]):
                for c in range(self.n_clusters[2]):
                    cluster_index = np.where((self.feature0[:, a] != 0) & (self.feature1[:, b] != 0) &
                                             (self.feature2[:, c] != 0))[0]  # 同时有这三个聚类结果的数
                    self.cluster_title[title] = [a, b, c]
                    if len(cluster_index) <= 5:
                        title += 1
                        # print("The number of cluster is less then 5, so [{},{},{}] abandon.".format(a, b, c))
                        continue
                    self.cocluster_membership[cluster_index, title] = self.feature0[cluster_index, a] * self.feature1[
                        cluster_index, b] * self.feature2[cluster_index, c]
                    title += 1

        # 三向隶属度归一化
        w = self.weight.reshape((-1, 1))
        self.cocluster_membership = self.cocluster_membership / self.cocluster_membership.sum(axis=1, keepdims=True)
        self.cocluster_membership = self.cocluster_membership * w

        # 计算每个聚类块的sc值
        s = []
        all_cluster = np.where(np.sum(self.cocluster_membership, axis=0) != 0)[0]  # 所有聚类（不包括空簇）
        if len(all_cluster) <= 1:
            return -1
        all_similar_mat = (similar_mat[:, :, 0] ** 2 + similar_mat[:, :, 1] ** 2 + similar_mat[:, :, 2] ** 2) ** (1/2)
        for cluster_id in all_cluster:  # 遍历每个簇
            if len(np.where(self.cocluster_membership[:, cluster_id] != 0)[0]) == 0:  # 该簇没有成员
                continue
            inner_membership_mat = self.cocluster_membership[:, cluster_id].reshape((-1, 1)) * \
                                   self.cocluster_membership[:, cluster_id].reshape((1, -1))
            for track_id in np.where(self.cocluster_membership[:, cluster_id] != 0)[0]:  # 遍历每个簇中的每个数据
                # 保存点到簇内距离的平均值
                dis = inner_membership_mat[track_id] * all_similar_mat[track_id]
                a = dis[dis != 0]
                a = np.sum(a) / np.sum(inner_membership_mat[track_id, dis != 0].flatten())  # 模糊平均簇内距离
                b = []  # 存放其他簇的平均模糊距离
                for other_cluster in all_cluster:
                    if other_cluster == cluster_id:
                        continue
                    out_membership_mat = self.cocluster_membership[:, other_cluster] * \
                                         self.cocluster_membership[track_id, cluster_id]
                    dis = out_membership_mat * all_similar_mat[track_id]  # 簇外距离
                    b.append(np.sum(dis[dis != 0]) / np.sum(out_membership_mat[dis != 0].flatten()))
                s.append((np.min(b) - a) / max(a, np.min(b)))
                # b = np.mean(all_similar_mat[track_id, np.where((dis == 0) & (np.sum(
                #     self.cocluster_membership, axis=1) != 0))[0]])  # 保存每个点到其他簇的平均距离
        print("SC of CoCluster:", np.nanmean(s))
        return np.nanmean(s)


class ConsensusThreeWaySpectralFuzzyClustering:  # 不用共聚类结果模糊SC
    def __init__(self, data_num, n_clusters, weight):
        self.data_num = data_num
        self.n_clusters = n_clusters  # 聚类数
        self.weight = weight  # 三向隶属度
        self.feature = np.zeros((data_num, n_clusters))
        self.estimate = -1  # 对每次聚类结果进行评估，越大越好
        self.cluster_title = {}  # 共聚类隶属度的表头标签（就是分别代表三个特征的聚类标签）

    def fit(self, Similar_matrix, Weight):  # 聚类
        similar_matrix = (Similar_matrix[:, :, 0] ** 2 + Similar_matrix[:, :, 1] ** 2 + Similar_matrix[:, :, 2] ** 2
                          ) ** (1/2)
        self.feature = self.spectralclustering(self.n_clusters, similar_matrix, Weight)

    def spectralclustering(self, n_clusters, similar_matrix, weight):
        w = np.zeros((similar_matrix.shape[0], similar_matrix.shape[0]), dtype=np.float64)
        for i in range(similar_matrix.shape[0]):
            for j in range(i, similar_matrix.shape[0]):
                w[j][i] = w[i][j] = weight[i] * weight[j]
        def AKNN(S, sigma=1.0):
            N = S.shape[0]
            A = np.zeros((N, N))
            for i in range(N):
                for j in range(i, N):
                    A[i][j] = np.exp(-S[i][j] ** 2 / 2 / sigma / sigma) * w[i, j]
                    A[j][i] = A[i][j]
            return A
        Adjacent = AKNN(similar_matrix, sigma=1.3)
        def calLaplacianMatrix(adjacentMatrix):
            # 计算标准化的拉普拉斯矩阵
            degreeMatrix = np.sum(adjacentMatrix, axis=1)
            laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix
            sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** 0.5))
            return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)
        Laplacian = calLaplacianMatrix(Adjacent)
        x, V = np.linalg.eig(Laplacian)
        x = zip(x, range(len(x)))
        x = sorted(x, key=lambda x: x[0])
        # 按列表的第一个元素(特征值)升序排列
        H = np.vstack([V[:, i] for (v, i) in x[:n_clusters]]).T  # 取最小的K个特征值对应的特征行向量,按垂直方向（行顺序）堆叠数组构成一个新的数组,然后转置

        # 模糊聚类
        membership_mat = np.random.random((len(H), n_clusters))  # 初始化随机隶属度
        membership_mat = np.divide(membership_mat, np.sum(membership_mat, axis=1)[:, np.newaxis])  # 归一化
        Weight = np.repeat(weight, n_clusters).reshape((-1, n_clusters))
        membership_mat = Weight * membership_mat
        while True:
            # 计算簇中心
            working_membership_mat = membership_mat ** 2
            Centroids = np.divide(np.dot(working_membership_mat.T, H),
                                  np.sum(working_membership_mat.T, axis=1)[:, np.newaxis])

            n_c_distance_mat = np.zeros((len(H), n_clusters))
            for i, x in enumerate(H):
                for j, c in enumerate(Centroids):
                    n_c_distance_mat[i][j] = np.linalg.norm(x - c, 2)

            new_membership_mat = np.zeros((len(H), n_clusters))

            for i, x in enumerate(H):
                for j, c in enumerate(Centroids):
                    # 通过之前的簇中心，更新u
                    new_membership_mat[i][j] = 1. / np.sum((n_c_distance_mat[i][j] / n_c_distance_mat[i]) ** 2)

            new_membership_mat = Weight * new_membership_mat
            for i in range(new_membership_mat.shape[0]):
                epsilon = 0
                for j in np.argsort(-new_membership_mat[i]):
                    if epsilon >= 0.8 * Weight[i][0] or new_membership_mat[i][j] < max(new_membership_mat[i]) / 4:
                        new_membership_mat[i][j] = 0
                    epsilon += new_membership_mat[i][j]
                # 数据归一化
                if sum(new_membership_mat[i]) != 0:
                    new_membership_mat[i] = new_membership_mat[i] / sum(new_membership_mat[i]) * Weight[i]
            if np.sum(abs(membership_mat - new_membership_mat)) < 0.0001:
                break
            membership_mat = new_membership_mat
        return membership_mat

    def evaluate(self, similar_mat):  # 对这次结果进行评估，并返回评估值
        sc = 0
        sc += self.cluster(self.feature, similar_mat[:, :, 0])  # 分别对三个特征和共聚类结果进行评估
        sc += self.cluster(self.feature, similar_mat[:, :, 1])
        sc += self.cluster(self.feature, similar_mat[:, :, 2])
        sc += self.co_cluster(similar_mat)
        self.estimate = sc / 4
        return sc / 4

    def cluster(self, feature, similar_mat):  # 分别根据不同的特征隶属度得到模糊SC，越大越好
        s = []
        all_cluster = list(range(self.n_clusters))  # 所有簇
        for cluster_id in all_cluster:  # 遍历每个簇
            inner_membership_mat = feature[:, cluster_id].reshape((-1, 1)) * feature[:, cluster_id].reshape((1, -1))
            for track_id in np.where(feature[:, cluster_id] != 0)[0]:  # 遍历每个簇中的每个数据
                # 保存点到簇内距离的平均值
                dis = inner_membership_mat[track_id] * similar_mat[track_id]
                a = dis[dis != 0]  # 簇内距离
                a = np.sum(a) / np.sum(inner_membership_mat[track_id, dis != 0].flatten())  # 模糊平均簇内距离
                b = []  # 存放其他簇的平均模糊距离
                for other_cluster in all_cluster:
                    if other_cluster == cluster_id:
                        continue
                    out_membership_mat = feature[:, other_cluster] * feature[track_id, cluster_id]
                    dis = out_membership_mat * similar_mat[track_id]  # 簇外距离
                    b.append(np.sum(dis[dis != 0]) / np.sum(out_membership_mat[dis != 0].flatten()))
                s.append((np.min(b) - a) / max(a, np.min(b)))
        print("SC of feature{}:", np.nanmean(s))
        return np.nanmean(s)

    def co_cluster(self, similar_mat):  # 共聚类评估结果模糊SC越大越好
        # 计算每个聚类块的sc值
        s = []
        all_cluster = np.where(np.sum(self.feature, axis=0) != 0)[0]  # 所有聚类（不包括空簇）
        if len(all_cluster) <= 1:
            return -1
        all_similar_mat = (similar_mat[:, :, 0] ** 2 + similar_mat[:, :, 1] ** 2 + similar_mat[:, :, 2] ** 2) ** (1/2)
        for cluster_id in all_cluster:  # 遍历每个簇
            if len(np.where(self.feature[:, cluster_id] != 0)[0]) == 0:  # 该簇没有成员
                continue
            inner_membership_mat = self.feature[:, cluster_id].reshape((-1, 1)) * \
                                   self.feature[:, cluster_id].reshape((1, -1))
            for track_id in np.where(self.feature[:, cluster_id] != 0)[0]:  # 遍历每个簇中的每个数据
                # 保存点到簇内距离的平均值
                dis = inner_membership_mat[track_id] * all_similar_mat[track_id]
                a = dis[dis != 0]
                a = np.sum(a) / np.sum(inner_membership_mat[track_id, dis != 0].flatten())  # 模糊平均簇内距离
                b = []  # 存放其他簇的平均模糊距离
                for other_cluster in all_cluster:
                    if other_cluster == cluster_id:
                        continue
                    out_membership_mat = self.feature[:, other_cluster] * \
                                         self.feature[track_id, cluster_id]
                    dis = out_membership_mat * all_similar_mat[track_id]  # 簇外距离
                    b.append(np.sum(dis[dis != 0]) / np.sum(out_membership_mat[dis != 0].flatten()))
                s.append((np.min(b) - a) / max(a, np.min(b)))
        print("SC of CoCluster:", np.nanmean(s))
        return np.nanmean(s)


class CCC:  # Consensus Co-Clustering
    def __init__(self, n_clusters, cluster_times=10, Noise_threshold=0.4):
        print("Start Consensus Co-Clustering.")
        self.n_clusters = n_clusters  # cluster number[0,0,0]分别针对不同特征的，目前固定三个特征值
        self.cluster_times = cluster_times  # cluster result , the number of population
        self.Noise_threshold = Noise_threshold  # 噪声阈值
        self.weight = None  # 三向权重

    def fit(self, similar_matrix, Weight):  # 训练模型
        x_num = similar_matrix.shape[0]  # 数量集的个数
        self.weight = Weight
        result = []  # 存放每次聚类结果
        result_estimate = []  # 每次结果评估
        print("The number of Cluster times is {}.".format(self.cluster_times))
        for t in range(self.cluster_times):
            print("The {}th clustering.".format(t))
            cluster_time = ConsensusThreeWaySpectralFuzzyClustering(data_num=x_num, n_clusters=self.n_clusters, weight=Weight)
            # 用三向谱模糊聚类得到隶属矩阵
            cluster_time.fit(Similar_matrix=similar_matrix, Weight=Weight)
            result_estimate.append(cluster_time.evaluate(similar_matrix))  # 对这次聚类进行评价
            result.append(cluster_time)
        print("Result Estimate of clustering:{}".format(result_estimate), "\nMax:{}".format(max(result_estimate)))


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

    W = []
    for i in range(len(track_list)):
        W.append(track_list[i].time_membership)
    W = np.array(W)  # 得到三向时间隶属度
    # CCC([3, 2, 3], cluster_times=5).fit(similar_matrix, W)
    CCC(3, cluster_times=5).fit(similar_matrix, W)
