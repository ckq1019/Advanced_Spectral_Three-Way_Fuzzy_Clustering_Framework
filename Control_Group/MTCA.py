import os
import time
import numpy as np
from Global_Variable import *


class MTCA:
    # 根据论文的MTCA算法进行轨迹聚类
    def __init__(self, num_track, file_path=None):
        self.num_track = num_track  # 轨迹数量
        self.run_time = 0  # 运行时间
        # self.Trajectory_dictionary = {}  # 轨迹字典{track_index: track_id}
        self.similar_mat = np.zeros((num_track, num_track))  # 相似度矩阵
        self.file_path = file_path  # 文件位置
        self.predict_labels = np.full(num_track, -1)  # 预测标签

    def MFHD(self, track_df, track_list, keep_file=False, read_file=False):
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

    def Tuning_parameters(self, eps, MinTRs):
        # 直接用OPTICS算法
        return

    def Cluster(self, eps=0, MinTRs=0):
        start_time = time.time()
        from sklearn.cluster import OPTICS
        cluster_result = OPTICS(metric='precomputed').fit_predict(self.similar_mat)
        end_time = time.time()
        self.run_time += (end_time - start_time)
        self.predict_labels = cluster_result

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
              np.nanmean(np.std(time_std, axis=0) / np.mean(time_std, axis=0)) / np.nanmean(np.nanmean(time_std, axis=1)))

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

    def keep_file(self, ):  # 存档
        np.save(os.path.join(self.file_path, "MTCA_run_time.npy"), self.run_time)
        np.save(os.path.join(self.file_path, "MTCA_similar_mat.npy"), self.similar_mat)
        np.save(os.path.join(self.file_path, "MTCA_Trajectory_dictionary.npy"), self.Trajectory_dictionary)
        print("Successfully saved progress.")

    def read_file(self):  # 读档
        self.run_time = np.float64(np.load(os.path.join(self.file_path, "MTCA_run_time.npy")))
        self.similar_mat = np.load(os.path.join(self.file_path, "MTCA_similar_mat.npy"))
        self.Trajectory_dictionary = np.load(os.path.join(self.file_path, "MTCA_Trajectory_dictionary.npy"),
                                             allow_pickle=True).item()
        print("Progress read successfully.")

    def Save_results(self, filepath):  # 结果保存
        result = []
        for i in range(self.num_track):
            result.append([self.Trajectory_dictionary[i], self.predict_labels[i]])
        np.save(os.path.join(filepath, "MTCA_result.npy"), result)
        print("Result saved successfully.")


if __name__ == '__main__':
    # Trajectory preprocessing
    file_path = r"cyclonic_[353  80].csv"
    from main import read_data
    origin_data, file_format = read_data(os.path.join(R"D:\Trajectory_analysis\Data", file_path))
    # 数据预处理
    from Data_Preprocessing import format_conversion
    data = format_conversion(origin_data, file_format=file_format, year_range=[2018, 2022], latitude_range=[0, 25.0],
                             longitude_range=[100.0, 122.0], Data_processing=False)
    track_list = []
    for i in data.keys():
        track_list.append(data[i])

    # start_time = time.time()
    K = 0
    from Similarity.Similarity_Measurement_Methodology import SimilarityMatrix
    sm = SimilarityMatrix(isCalculate=False, TR_List=track_list, K=K, filepath=os.path.join(os.getcwd(), "output",
                                                                                            'HD_Similarity_matrix.npy'))
    similaritymatrix = sm.similarity_matrix[:, :, 0]
    # from KNN.KDtree import KNNTree
    # k = KNNTree(track_list, None)
    # cal_type = 'P'  # 'P' or 'L'
    # for track_index in range(len(track_list)):
    #     sm.calculation(track_index, k.knn(track_index, K), cal_type, 'HD', Compression=None)
    # sm.savefile(path=os.path.join(SIMILAR_MATRIX, 'HD_Similarity_matrix.npy'))
    # end_time = time.time()
    # print("耗时：", end_time - start_time)
    # DBSCAN
    # from Cluster.DBSCAN import DBSCAN
    # result = DBSCAN(track_list, 0.21330, 2, similar_matrix=similaritymatrix)
    # print("簇的数量为：", len(result.keys()))
    # # 结果评估
    # result_analysis = []
    # cluster_std = np.zeros((len(result.keys()), 2))  # 记录每个簇的时间戳初末的方差
    # for i, cluster_id in enumerate(result.keys()):
    #     timespan = []
    #     for track_id in result[cluster_id]:
    #         from Similarity.Similarity_Measurement_Methodology import time_change
    #         start_days = time_change(data[track_id].TimeSpan[0])
    #         end_days = start_days + (data[track_id].TimeSpan[1] - data[track_id].TimeSpan[0])
    #         timespan.append([start_days, end_days])  # 获得每个轨迹的时间戳
    #     timespan = np.array(timespan, dtype=int)
    #     if len(np.where(timespan[:, 0] > (365 / 3 * 2))[0]) != 0 and len(
    #             np.where(timespan[:, 1] < (365 / 3))[0]) != 0:
    #         # 如果同时存在年末和年初的轨迹则需要处理
    #         if len(np.where(timespan[:, 0] > (365 / 3 * 2))[0]) > len(np.where(timespan[:, 1] < (365 / 3))[0]):
    #             timespan[timespan[:, 0] < (365 / 3 * 2)] = timespan[timespan[:, 0] < (365 / 3 * 2)] + 365
    #         else:
    #             timespan[timespan[:, 0] > (365 / 3)] = timespan[timespan[:, 0] > (365 / 3)] - 365
    #
    #     cluster_std[i] = np.std(timespan, axis=0) / np.mean(timespan, axis=0)  # 轨迹出发点和结束点时间戳方差
    #
    # print("时间戳初末的CV奇异系数（越大越好）：", np.nanmean(np.std(cluster_std, axis=0) / np.mean(cluster_std, axis=0)) /
    #       np.nanmean(np.nanmean(cluster_std, axis=1)))
    # result_analysis.append(np.nanmean(np.std(cluster_std, axis=0) / np.mean(cluster_std, axis=0)) /
    #                        np.nanmean(np.nanmean(cluster_std, axis=1)))
    # # 2.分析初末点
    # cluster_std = np.zeros((len(result.keys()), 2))  # 记录每个簇的初末点的方差
    # for i, cluster_id in enumerate(result.keys()):
    #     start_point_matrix = []
    #     end_point_matrix = []
    #     for track_id in result[cluster_id]:
    #         start_point_matrix.append(data[track_id].point_list[0].origin_array()[3:5])
    #         end_point_matrix.append(data[track_id].point_list[-1].origin_array()[3:5])
    #     # 数据标准化
    #     start_point_matrix = np.array(start_point_matrix, dtype=np.float64)
    #     end_point_matrix = np.array(end_point_matrix, dtype=np.float64)
    #     cluster_std[i] = [np.nanmean(np.std(start_point_matrix, axis=0) / np.mean(start_point_matrix, axis=0)
    #                                  ),
    #                       np.nanmean(np.std(end_point_matrix, axis=0) / np.mean(end_point_matrix, axis=0)
    #                                  )]  # 轨迹出发点时间戳方差
    # print("初末点的CV奇异系数（越大越好）：", np.nanmean(np.std(cluster_std, axis=0) / np.mean(cluster_std, axis=0)) /
    #       np.nanmean(np.nanmean(cluster_std, axis=1)))
    # result_analysis.append(np.nanmean(np.std(cluster_std, axis=0) / np.mean(cluster_std, axis=0)) /
    #                        np.nanmean(np.nanmean(cluster_std, axis=1)))
    #
    # # 2.分析变化趋势
    # name = ["TrackId", "timestamp", "amplitude", "x_coordinate", "y_coordinate", "speed_average",
    #         "effective_radius",
    #         "velocity", "angle"]
    # all_corr_matrix = []  # 每个聚类的所有轨迹的协方差的平均值
    # all_mean_corr_matrix = []  # 存放每个簇的偏相关性的平均值
    # from Data_Analysis.Data_Analysis import correlation
    # for i, cluster_id in enumerate(result.keys()):  # 遍历每个集群
    #     corr_matrix = []  # 每条协方差矩阵
    #     for track_id in result[cluster_id]:  # 遍历每条轨迹
    #         point_matrix = []
    #         for Point in data[track_id].point_list:  # 遍历每个轨迹点
    #             point_matrix.append(Point.origin_array())
    #         point_matrix = np.array(point_matrix, dtype=np.float64)
    #         corr_matrix.append(correlation(point_matrix, method='pearson'))  # 分析协方差
    #     corr_matrix = np.array(corr_matrix, dtype=np.float64)
    #     all_corr_matrix.append(np.std(corr_matrix, axis=0))  # 所有的轨迹的标准差
    #     all_mean_corr_matrix.append(np.nanmean(corr_matrix, axis=0))  # 簇内的轨迹的标准差
    # all_corr_matrix = np.array(all_corr_matrix, dtype=np.float64)
    # all_mean_corr_matrix = np.array(all_mean_corr_matrix, dtype=np.float64)
    #
    # print("变化趋势（越大越好）：", np.nanmean(np.std(all_mean_corr_matrix, axis=0)) /
    #       np.nanmax(np.nanmean(all_corr_matrix, axis=1)))
    # result_analysis.append(np.nanmean(np.std(all_mean_corr_matrix, axis=0)) /
    #                        np.nanmax(np.nanmean(all_corr_matrix, axis=1)))
