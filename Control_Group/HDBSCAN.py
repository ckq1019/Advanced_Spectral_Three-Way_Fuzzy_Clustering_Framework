import time
import numpy as np
from Global_Variable import *


class Hdbscan:
    # 根据论文的HDBSCAN算法进行轨迹聚类
    def __init__(self, num_track, file_path=None):
        self.num_track = num_track  # 轨迹数量
        self.run_time = 0  # 运行时间
        self.similar_mat = np.zeros((num_track, num_track))  # 相似度矩阵
        self.file_path = file_path  # 文件位置
        self.predict_labels = np.full(num_track, -1)  # 预测标签

    def HD(self, track_df, track_list, keep_file=False, read_file=False):
        if read_file:
            self.read_file()
        else:
            from Similarity.HD import MFHD
            start_time = time.time()
            for i, track_a in enumerate(track_list):
                for j, track_b in enumerate(track_list):
                    if track_a == track_b or self.similar_mat[i][j] != 0:
                        continue
                    self.similar_mat[i][j] = self.similar_mat[j][i] = max(MFHD(
                        track_df[track_df["track_id"] == track_a][["x", "y", "v", "a"]],
                        track_df[track_df["track_id"] == track_b][["x", "y", "v", "a"]], cal_type='Point'),
                        MFHD(
                            track_df[track_df["track_id"] == track_b][["x", "y", "v", "a"]],
                            track_df[track_df["track_id"] == track_a][["x", "y", "v", "a"]], cal_type='Point')
                    )
            end_time = time.time()
            self.run_time += (end_time - start_time)
            if keep_file:
                self.keep_file()  # 存档

    def Cluster(self, eps=0, MinTRs=0):
        start_time = time.time()
        from sklearn.cluster import HDBSCAN
        cluster_result = HDBSCAN(min_cluster_size=10, metric='precomputed').fit_predict(self.similar_mat)
        end_time = time.time()
        self.run_time += (end_time - start_time)
        self.predict_labels = cluster_result
