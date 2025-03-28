import os
import numpy as np
import random
import math


class CLUSTER:  # 保存每次聚类结果
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

    def evaluate(self, similar_mat, hard=False):  # 对这次结果进行评估，并返回评估值
        sc = 0
        if hard:  # 硬聚类评估
            sc += self.hard_cluster([0], similar_mat[:, :, 0])
            sc += self.hard_cluster([1], similar_mat[:, :, 1])
            sc += self.hard_cluster([2], similar_mat[:, :, 2])
            sc += self.hard_cluster([0, 1, 2], similar_mat)
        else:
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

        for row in range(self.cocluster_membership.shape[0]):
            epsilon = 0
            for c in np.argsort(-self.cocluster_membership[row]):
                if epsilon > 0.8 * self.weight[row] or self.cocluster_membership[row][c] < max(
                        self.cocluster_membership[row]) / 4:
                    self.cocluster_membership[row][c] = 0
                epsilon += self.cocluster_membership[row][c]
            if sum(self.cocluster_membership[row]) != 0:
                self.cocluster_membership[row] = self.cocluster_membership[row] / sum(
                    self.cocluster_membership[row]) * self.weight[row]
        for col in range(self.cocluster_membership.shape[1]):  # 遍历每个簇的如果该簇的核心轨迹数量少于2，则废弃
            if len(np.where(self.cocluster_membership[:, col] == 1)[0]) < 2:
                self.cocluster_membership[:, col] = 0

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

    def hard_cluster(self, feature_index, similar_mat):  # 对核心（硬聚类）进行评估
        if len(feature_index) == 1:  # 对单个特征进行评估
            if feature_index[0] == 0:
                feature = self.feature0
            elif feature_index[0] == 1:
                feature = self.feature1
            elif feature_index[0] == 2:
                feature = self.feature2
            s = []
            all_cluster = list(range(self.n_clusters[feature_index[0]]))  # 所有簇
            for cluster_id in all_cluster:  # 遍历每个簇
                for track_id in np.where(feature[:, cluster_id] == 1)[0]:  # 遍历每个簇中的每个数据
                    # 保存点到簇内距离的平均值
                    dis = similar_mat[track_id, np.where(feature[:, cluster_id] == 1)[0]]
                    a = dis[dis != 0]  # 簇内距离
                    if len(a) <= 1:
                        continue
                    a = np.mean(a)
                    b = []  # 存放其他簇的平均模糊距离
                    for other_cluster in all_cluster:
                        if other_cluster == cluster_id:
                            continue
                        dis = similar_mat[track_id, np.where(feature[:, other_cluster] == 1)[0]]  # 簇外距离
                        b.append(np.nanmean(dis[dis != 0]))
                    s.append((np.min(b) - a) / max(a, np.min(b)))
            print("SC of feature{}:".format(feature_index[0]), np.nanmean(s))
            return np.nanmean(s)
        else:
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
                        self.cocluster_membership[cluster_index, title] = self.feature0[cluster_index, a] * \
                                                                          self.feature1[
                                                                              cluster_index, b] * self.feature2[
                                                                              cluster_index, c]
                        title += 1

            # 三向隶属度归一化
            w = self.weight.reshape((-1, 1))
            self.cocluster_membership = self.cocluster_membership / self.cocluster_membership.sum(axis=1, keepdims=True)
            self.cocluster_membership = self.cocluster_membership * w

            for row in range(self.cocluster_membership.shape[0]):
                epsilon = 0
                for c in np.argsort(-self.cocluster_membership[row]):
                    if epsilon > 0.8 * self.weight[row] or self.cocluster_membership[row][c] < max(
                            self.cocluster_membership[row]) / 4:
                        self.cocluster_membership[row][c] = 0
                    epsilon += self.cocluster_membership[row][c]
                if sum(self.cocluster_membership[row]) != 0:
                    self.cocluster_membership[row] = self.cocluster_membership[row] / sum(
                        self.cocluster_membership[row]) * self.weight[row]
            for col in range(self.cocluster_membership.shape[1]):  # 遍历每个簇的如果该簇的核心轨迹数量少于2，则废弃
                if len(np.where(self.cocluster_membership[:, col] == 1)[0]) < 2:
                    self.cocluster_membership[:, col] = 0

            # 计算每个聚类块的sc值
            s = []
            all_cluster = np.where(np.sum(self.cocluster_membership, axis=0) != 0)[0]  # 所有聚类（不包括空簇）
            if len(all_cluster) <= 1:
                return -1
            all_similar_mat = (similar_mat[:, :, 0] ** 2 + similar_mat[:, :, 1] ** 2 + similar_mat[:, :, 2] ** 2) ** (
                        1 / 2)
            for cluster_id in all_cluster:  # 遍历每个簇
                if len(np.where(self.cocluster_membership[:, cluster_id] == 1)[0]) == 0:  # 该簇没有成员
                    continue
                for track_id in np.where(self.cocluster_membership[:, cluster_id] == 1)[0]:  # 遍历每个簇中的每个数据
                    # 保存点到簇内距离的平均值
                    dis = all_similar_mat[track_id, np.where(self.cocluster_membership[:, cluster_id] == 1)[0]]
                    a = dis[dis != 0]
                    a = np.nanmean(a)
                    b = []  # 存放其他簇的平均模糊距离
                    for other_cluster in all_cluster:
                        if other_cluster == cluster_id or len(np.where(self.cocluster_membership[:,
                                                                       other_cluster] == 1)[0]) == 0:
                            continue
                        dis = all_similar_mat[track_id, np.where(self.cocluster_membership[:, other_cluster] == 1)[0]]  # 簇外距离
                        b.append(np.nanmean(dis[dis != 0]))
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

    def fit(self, similar_matrix, Weight, SAVE=True):  # 训练模型
        x_num = similar_matrix.shape[0]  # 数量集的个数
        self.weight = Weight
        result = []  # 存放每次聚类结果
        result_estimate = []  # 每次结果评估
        print("The number of Cluster times is {}.".format(self.cluster_times))
        from Cluster.SC import mySpectral_Clustering
        for t in range(self.cluster_times):
            print("The {}th clustering.".format(t))
            cluster_time = CLUSTER(data_num=x_num, n_clusters=self.n_clusters, weight=Weight)
            # 用三向谱模糊聚类得到隶属矩阵
            cluster_time.feature0 = mySpectral_Clustering(n_clusters=self.n_clusters[0],
                                                          Similar_matrix=similar_matrix[:, :, 0], Weight=Weight)
            cluster_time.feature1 = mySpectral_Clustering(n_clusters=self.n_clusters[1],
                                                          Similar_matrix=similar_matrix[:, :, 1], Weight=Weight)
            cluster_time.feature2 = mySpectral_Clustering(n_clusters=self.n_clusters[2],
                                                          Similar_matrix=similar_matrix[:, :, 2], Weight=Weight)
            result_estimate.append(cluster_time.evaluate(similar_matrix, hard=False))  # 对这次聚类进行评价
            result.append(cluster_time)
        print("Result Estimate of clustering:{}".format(result_estimate), "\nMax:{}".format(max(result_estimate)))
        if SAVE:
            file_path = r"D:\Trajectory_analysis\Cluster\output"
            np.save(os.path.join(file_path, "feature0.npy"), result[np.argmax(result_estimate)].feature0)
            np.save(os.path.join(file_path, "feature1.npy"), result[np.argmax(result_estimate)].feature1)
            np.save(os.path.join(file_path, "feature2.npy"), result[np.argmax(result_estimate)].feature2)
            np.save(os.path.join(file_path, "cocluster.npy"), result[np.argmax(result_estimate)].cocluster_membership)
            print("Cluster Over!\nThe best result of clustering is saved successful.")


if __name__ == '__main__':
    file_path = r"Eddy_trajectory_nrt_3.2exp_cyclonic_20180101_20220210.csv"
    from main import read_data
    origin_data, file_format = read_data(os.path.join(r'D:\Trajectory_analysis\Data', file_path))
    from Data_Preprocessing import format_conversion
    data = format_conversion(origin_data, file_format='CSV', Data_processing=False)
    track_list = []
    for i in data.keys():
        track_list.append(data[i])
    from Similarity.Similarity_Measurement_Methodology import SimilarityMatrix
    sm = SimilarityMatrix(isCalculate=False, TR_List=track_list, K=0, reshuffle=False,
                          filepath=os.path.join(r'D:\Trajectory_analysis\Similarity\output', 'All_Similarity_matrix.npy'))
    similar_matrix = sm.similarity_matrix

    W = np.full(len(track_list), 1)
    # for i in range(len(track_list)):
    #     W.append(track_list[i].time_membership)
    # W = np.array(W)  # 得到三向时间隶属度
    CCC([4, 3, 4]).fit(similar_matrix, W, SAVE=False)
    # np.save(os.path.join(r"D:\Trajectory_analysis\Cluster\output", "cocluster_membership.npy"),
    #         cluster_result.cocluster_membership)
    # print("{} file saved successful.".format(os.path.join(r"D:\Trajectory_analysis\Cluster\output",
    #                                                       "cocluster_membership.npy")))

