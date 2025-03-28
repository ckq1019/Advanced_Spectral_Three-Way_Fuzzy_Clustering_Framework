import os
import numpy as np
import time


class ISCM:
    # 根据论文的ISCM算法进行轨迹聚类
    def __init__(self, num_track, file_path=None):
        self.num_track = num_track  # 轨迹数量
        self.run_time = 0  # 运行时间
        # self.Trajectory_dictionary = {}  # 轨迹字典{track_index: track_id}
        self.similar_mat = np.zeros((num_track, num_track))  # 相似度矩阵
        self.file_path = file_path  # 文件位置
        self.predict_labels = np.full(num_track, -1)  # 预测标签

    def Cluster(self, n_clusters, track_df, track_id, keep_file=False, read_file=False):
        if read_file:
            self.read_file()
        else:
            # ADPS轨迹压缩，相似度度量:DTW
            # track_id = list(track_df["track_id"].unique())
            start_time = time.time()
            for i, track_a in enumerate(track_id):  # 遍历轨迹
                # self.Trajectory_dictionary[i] = track_a  # 轨迹id
                for j, track_b in enumerate(track_id):
                    if track_a == track_b or self.similar_mat[i][j] != 0:
                        continue
                    self.similar_mat[i][j] = self.similar_mat[j][i] = self.similarity(
                        track_df[track_df["track_id"] == track_a][["x", "y"]],
                        track_df[track_df["track_id"] == track_b][["x", "y"]])
            # 相似度矩阵进行转换
            for i in range(self.similar_mat.shape[0]):
                for j in range(self.similar_mat.shape[1]):
                    if self.similar_mat[i][j] == 0:
                        continue
                    self.similar_mat[i][j] = 1 / self.similar_mat[i][j]
            from sklearn.cluster import SpectralClustering
            self.predict_labels = SpectralClustering(n_clusters=n_clusters, affinity="precomputed").fit_predict(
                self.similar_mat)
            end_time = time.time()
            self.run_time += (end_time - start_time)
            if keep_file:
                self.keep_file()  # 存档

    def similarity(self, track_a, track_b):
        track_a = np.array(track_a, dtype=np.float64)
        track_b = np.array(track_b, dtype=np.float64)
        from Similarity.DTW import DTW
        return DTW(track_a, track_b, windowSize=0)

    def Evaluation(self, track_df):
        clusters = list(set(self.predict_labels))  # 簇标签
        time_std = np.zeros((len(clusters), 2))  # 记录每个簇的时间戳初末的方差
        for i, cluster_id in enumerate(clusters):  # 遍历每个簇
            timespan = []  # 存储每个簇的时间跨度[[start0, end0],[start1, end1]]
            for track_idx in np.where(self.predict_labels == cluster_id)[0]:
                track_id = self.Trajectory_dictionary[track_idx]
                start_days = track_df[track_df["track_id"] == track_id]["time"].min()
                end_days = track_df[track_df["track_id"] == track_id]["time"].max()
                timespan.append([start_days, end_days])  # 获得每个轨迹的时间戳
            timespan = np.array(timespan, dtype=int)
            time_std[i] = np.std(timespan, axis=0) / np.mean(timespan, axis=0)  # 轨迹出发点和结束点时间戳方差
        print("时间戳初末的CV奇异系数（越大越好）：",
              np.nanmean(np.std(time_std, axis=0) / np.mean(time_std, axis=0)) / np.nanmean(
                  np.nanmean(time_std, axis=1)))

        se_std = np.zeros((len(clusters), 2))  # 记录每个簇的初末点的方差
        for i, cluster_id in enumerate(clusters):
            start_point_matrix = []
            end_point_matrix = []
            for track_idx in np.where(self.predict_labels == cluster_id)[0]:
                track_id = self.Trajectory_dictionary[track_idx]
                start_point_matrix.extend(np.array(track_df[track_df["track_id"] == track_id][["x", "y"]]).tolist())
                end_point_matrix.extend(np.array(track_df[track_df["track_id"] == track_id][["x", "y"]]).tolist())
            # 数据标准化
            start_point_matrix = np.array(start_point_matrix, dtype=np.float64)
            end_point_matrix = np.array(end_point_matrix, dtype=np.float64)
            se_std[i] = [np.nanmean(np.std(start_point_matrix, axis=0) / np.mean(start_point_matrix, axis=0)),
                         np.nanmean(np.std(end_point_matrix, axis=0) / np.mean(end_point_matrix, axis=0))]
        print("初末点的CV奇异系数（越大越好）：", np.nanmean(np.std(se_std, axis=0) / np.mean(se_std, axis=0)) /
              np.nanmean(np.nanmean(se_std, axis=1)))

        all_corr_matrix = []  # 每个聚类的所有轨迹的协方差的平均值
        all_mean_corr_matrix = []  # 存放每个簇的偏相关性的平均值
        for i, cluster_id in enumerate(clusters):  # 遍历每个集群
            corr_matrix = []  # 每条协方差矩阵
            for track_idx in np.where(self.predict_labels == cluster_id)[0]:
                track_id = self.Trajectory_dictionary[track_idx]
                point_matrix = np.array(track_df[track_df["track_id"] == track_id][["time", "x", "y", "v", "a"]],
                                        dtype=np.float64)
                corr_matrix.append(np.corrcoef(point_matrix.T)[0])  # 分析协方差
            corr_matrix = np.array(corr_matrix, dtype=np.float64)
            all_corr_matrix.append(np.std(corr_matrix, axis=0))  # 所有的轨迹的标准差
            all_mean_corr_matrix.append(np.nanmean(corr_matrix, axis=0))  # 簇内的轨迹的标准差
        all_corr_matrix = np.array(all_corr_matrix, dtype=np.float64)
        all_mean_corr_matrix = np.array(all_mean_corr_matrix, dtype=np.float64)
        print("变化趋势（越大越好）：", np.nanmean(np.std(all_mean_corr_matrix, axis=0)) /
              np.nanmax(np.nanmean(all_corr_matrix, axis=1)))

        labels = []  # [true, predict]
        for i in range(self.num_track):
            labels.append([int(track_df[track_df["track_id"] == self.Trajectory_dictionary[i]]["label"].unique()),
                           self.predict_labels[i]])
        labels = np.array(labels)
        from sklearn.metrics import accuracy_score
        print("ACC:", accuracy_score(labels[:, 0], labels[:, 1]))

        print("Run time:", self.run_time)

    def keep_file(self):  # 存档
        np.save(os.path.join(self.file_path, "ISCM_run_time.npy"), self.run_time)
        np.save(os.path.join(self.file_path, "ISCM_similar_mat.npy"), self.similar_mat)
        np.save(os.path.join(self.file_path, "ISCM_predict_labels.npy"), self.predict_labels)
        np.save(os.path.join(self.file_path, "ISCM_Trajectory_dictionary.npy"), self.Trajectory_dictionary)
        print("Successfully saved progress.")

    def read_file(self):  # 读档
        self.run_time = np.float64(np.load(os.path.join(self.file_path, "ISCM_run_time.npy")))
        self.similar_mat = np.load(os.path.join(self.file_path, "ISCM_similar_mat.npy"))
        self.predict_labels = np.load(os.path.join(self.file_path, "ISCM_predict_labels.npy"))
        self.Trajectory_dictionary = np.load(os.path.join(self.file_path, "ISCM_Trajectory_dictionary.npy"),
                                             allow_pickle=True).item()
        print("Progress read successfully.")

