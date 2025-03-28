import netCDF4
import re
import pandas as pd
from Global_Variable import *
from Data_Preprocessing import format_conversion
from Data_Analysis.Data_Visualization import ModelVisualization
import os
from Data_Analysis.Data_Analysis import Similarity_analysis
from Similarity.Similarity_Measurement_Methodology import SimilarityMatrix
from Dimension_Reduction.Dimension_Reduction_Methodology import SimilarMatrixProjection
from Cluster.Trajectory_Clustering_Methodology import ClusterModel
from KNN.R_TREE import RTree
from KNN.KDtree import KNNTree
from Intensive_Time_Extraction.Time_Extraction import Intensive_time_interval
from Intensive_Space_Extraction.Sapce_Extraction import Intensive_space_interval


def read_data(file):
    '''
    :param file: 输入文件格式
    :return: 如果是txt 返回每行的字符串，以列表形式返回,如果是nc文件的话就返回字典
    '''
    if file.endswith('.txt'):
        if not os.path.exists(file):
            print("file not exists!")
        with open(file) as f:
            content = f.readlines()
        return content, 'TXT'
    elif file.endswith('.csv'):
        if not os.path.exists(file):
            print("file not exists!")
        content = pd.read_csv(file)
        filename = os.path.basename(file)
        if filename.startswith("cyclonic_"):  # 读取的是密度时间间隔提取的文件
            time_scope = re.compile(r'[\d+ \d+]').search(file).group()
            del_col = []
            for column in content.columns:  # 遍历每一列名
                t = re.compile(r'[\d+ \d+]').search(column)
                if t is not None and t.group() != time_scope:
                    del_col.append(column)
            content = content.drop(labels=del_col, axis=1)
        # with open(file, 'r') as file:
        #     content = file.readlines()
        # return content, 'TXT'
        return content, 'CSV'
    elif file.endswith('.nc'):
        if not os.path.exists(file):
            print("file not exists!")
        content = netCDF4.Dataset(file)
        return content.variables, 'NC'


def result_integration(all_result, result_cluster, tr_list):  # 把结果进行整合
    for center_id in result_cluster:  # 遍历每个簇
        if center_id == -1 or center_id == 'noise':
            continue
        # all_result[tr_list[center_id].TrackId] = []
        all_result[center_id] = []
        for tr_index in result_cluster[center_id]:
            # all_result[tr_list[center_id].TrackId].append(tr_list[tr_index].TrackId)
            all_result[center_id].append(tr_list[tr_index].TrackId)
            tr_list[tr_index].ClusterId = center_id


if __name__ == '__main__':
    # 数据预处理
    # file_path = r"Eddy_trajectory_nrt_3.2exp_cyclonic_20180101_20220210.nc"
    file_path = r"Eddy_trajectory_nrt_3.2exp_cyclonic_20180101_20220210.csv"
    # file_path = r"cyclonic_[353  80].csv"
    # file_path = r"hurdat2-1851-2021-100522.txt"
    origin_data, file_format = read_data(os.path.join(DATA_PATH, file_path))
    # 数据预处理
    data = format_conversion(origin_data, file_format=file_format, year_range=[2018, 2022], latitude_range=[0, 25.0],
                             longitude_range=[100.0, 122.0], Data_processing=False)
    track_list = []
    for i in data.keys():
        track_list.append(data[i])

    membership_mat = Intensive_time_interval(track_list, Show=True, Save=False)

    PREPROSS = False  # 数据是否处理过后（TRUE：直接读文件中的相似度矩阵，FALSE:需要先对数据进行计算）
    K = 0
    sm = SimilarityMatrix(isCalculate=PREPROSS, TR_List=track_list, K=K, reshuffle=False,
                          filepath=os.path.join(SIMILAR_MATRIX, 'Similarity_matrix.npy'))
    if PREPROSS:
        # reshuffle = False  # 对Rtree进行刷新
        # rtree = RTree(reshuffle=reshuffle)  # 构建Rtree
        # if reshuffle:
        #     for track_index, track in enumerate(track_list):
        #         rtree.insert(track_index, track)  # 插入叶节点
        # 检索KNN轨迹
        k = KNNTree(track_list, membership_mat)
        cal_type = 'L'  # 'P' or 'L'
        for track_index in range(len(track_list)):
            # feature_matrix = []
            # for i in data.keys():
            #     feature_matrix.extend(
            #         trajectory_segment(data[i], draw=False, algorithm='My_Method', cal_type=cal_type,
            #                            PCA_precess=False))
            # feature_matrix = np.array(feature_matrix, dtype=np.float64)
            # if feature_matrix.shape[1] < 9:
            #     for i in range(2, feature_matrix.shape[1]):
            #         if (max(feature_matrix[:, i]) - min(feature_matrix[:, i])) != 0:
            #             feature_matrix[:, i] = (feature_matrix[:, i] - min(feature_matrix[:, i])) / (max(
            #                 feature_matrix[:, i]) - min(feature_matrix[:, i]))
            #     if cal_type == 'L':
            #         df = pd.DataFrame(data=feature_matrix, columns=['TrackId', 'LineId', 'asr_v', 'x_v', 'y_v',
            #                                                         'velocity_v', 'angle_v'])
            #     elif cal_type == 'P':
            #         df = pd.DataFrame(data=feature_matrix, columns=['TrackId', 'time', 'asr', 'x', 'y',
            #                                                         'velocity', 'angle'])
            # else:
            #     # 对特征进行归一化,方便计算
            #     for i in range(2, feature_matrix.shape[1]):
            #         if (max(feature_matrix[:, i]) - min(feature_matrix[:, i])) != 0:
            #             feature_matrix[:, i] = (feature_matrix[:, i] - min(feature_matrix[:, i])) / (
            #                     max(feature_matrix[:, i]) -
            #                     min(feature_matrix[:, i]))
            #     if cal_type == 'L':
            #         df = pd.DataFrame(data=feature_matrix,
            #                           columns=['TrackId', 'LineId', 'amplitude_v', 'x_v', 'y_v',
            #                                    'speed_v', 'radius_v', 'velocity_v', 'angle_v'])
            #     elif cal_type == 'P':
            #         df = pd.DataFrame(data=feature_matrix, columns=['TrackId', 'time', 'amplitude', 'x', 'y',
            #                                                         'speed', 'radius', 'velocity', 'angle'])
            # sm.calculation(track_index, rtree.knn(track_list[track_index], K, track_list),
            #                cal_type, 'DTW', Compression='WT')
            # sm.calculation(track_index, k.knn(track_index, K), cal_type, 'DTW', Compression='WT')
            sm.calculation(track_index, k.knn(track_index, K), cal_type, 'DTW', Compression=None)
        sm.savefile()
            # similar_matrix = Similarity_Matrix(track_list, cal_type=cal_type, isCalculate=PREPROSS, algorithm='DTW',
            #                                    Compression=True)
    else:
        similar_matrix = sm.similarity_matrix
        # similar_matrix = Similarity_Matrix(track_list, cal_type=None, isCalculate=PREPROSS, algorithm=None)
    all_result = {}  # 所有的结果聚类：result{center_id:[track_id,...], ...}
    # 根据相似度矩阵转换成点聚类[[变化趋势*1],[初末位置*1],[时间戳相似度*1],...]
    point_matrix = SimilarMatrixProjection(algorithm='MDS', n_components=1, reshuffle=False, filepath=os.path.join(
        POINT_MATRIX, "knn_point_matrix.npy")).fit(similar_matrix)
    # model = ['FCM']  # 'AP', 'KMeans', 'SC', 'OPTICS', 'DPC', 'FCM', 'HC'
    # result_cluster = ClusterModel(model, point_matrix, K=20, similar_matrix=similar_matrix, draw=False, save=False,
    #                               denoise=False)
    # # 单个变量
    # result_cluster = {-1: [82, 15, 33, 40, 227, 129, 136, 193, 85, 179, 234, 61, 156, 189], 0: [16, 21, 34, 41, 45, 49, 53, 71, 74, 94, 99, 102, 103, 116, 130, 131, 135, 153, 163, 165, 170, 182, 186, 188, 195, 203, 221], 1: [7, 9, 13, 18, 25, 27, 32, 35, 48, 54, 63, 76, 81, 86, 87, 91, 92, 96, 111, 118, 123, 141, 148, 149, 154, 158, 167, 173, 180, 183, 194, 197, 201, 209, 219, 223, 226, 229], 2: [36, 51, 66, 117, 125, 164, 172, 211, 225, 228], 3: [37, 67, 88, 105, 114, 132, 142, 171, 185, 231], 5: [6, 14, 22, 24, 30, 39, 60, 83, 108, 113, 119, 128, 134, 161, 184, 190, 207, 213, 232], 7: [3, 8, 19, 26, 31, 42, 58, 62, 73, 104, 110, 124, 133, 146, 150, 174, 181, 196, 198, 212, 233], 8: [0, 4, 10, 11, 23, 28, 38, 43, 47, 52, 57, 59, 65, 70, 80, 84, 90, 98, 100, 101, 106, 115, 121, 122, 137, 138, 143, 144, 145, 155, 159, 176, 178, 191, 199, 200, 202, 204, 205, 208, 214, 218, 220], 16: [2, 5, 69, 75, 78, 95, 120, 126, 127, 147, 151, 162, 168, 169, 187, 210, 216, 222, 230], 22: [1, 12, 17, 20, 29, 44, 46, 50, 55, 56, 64, 68, 72, 77, 79, 89, 93, 97, 107, 109, 112, 139, 140, 152, 157, 160, 166, 175, 177, 192, 206, 215, 217, 224]}
    result_cluster = {-1: [67, 68, 206, 224, 263, 320, 389], 0: [20, 25, 26, 40, 42, 44, 47, 50, 62, 64, 65, 66, 131, 137, 143, 145, 151, 156, 160, 161, 165, 167, 171, 173, 257, 258, 260, 267, 271, 276, 277, 287, 353, 358, 359, 360, 366, 373, 374, 375, 378, 380, 390], 1: [78, 89, 94, 95, 102, 103, 108, 110, 111, 115, 186, 194, 197, 198, 204, 205, 209, 214, 220, 225, 232, 296, 302, 308, 311, 313, 315, 316, 322, 323, 327, 331, 332, 393, 394, 400, 402, 407, 411, 414, 427, 429, 430, 433, 443], 2: [7, 8, 11, 13, 14, 16, 23, 24, 28, 31, 33, 36, 38, 122, 129, 130, 132, 134, 135, 144, 159, 218, 235, 238, 239, 244, 246, 247, 249, 252, 255, 256, 266, 288, 345, 350, 354, 355, 361, 362, 364, 365, 370, 376, 447, 449, 458, 459], 3: [34, 41, 56, 138, 146, 154, 157, 158, 163, 166, 168, 172, 259, 269, 270, 274, 368, 369, 371, 377, 386, 387, 396], 4: [1, 2, 3, 6, 12, 19, 70, 98, 99, 104, 107, 121, 123, 125, 184, 187, 226, 229, 240, 245, 289, 293, 295, 306, 326, 329, 333, 334, 336, 339, 341, 342, 344, 351, 392, 415, 431, 432, 435, 438, 441, 448, 451, 452, 454, 460], 5: [106, 113, 114, 117, 119, 180, 188, 212, 222, 223, 228, 230, 231, 233, 291, 297, 317, 319, 324, 328, 330, 337, 338, 410, 413, 416, 424, 425, 426, 439, 442, 444], 6: [4, 5, 10, 15, 17, 21, 22, 29, 112, 116, 120, 124, 126, 127, 128, 133, 136, 139, 179, 215, 221, 236, 237, 241, 243, 248, 250, 254, 281, 343, 347, 348, 349, 357, 422, 450, 455, 456], 7: [27, 77, 170, 181, 262, 278, 279, 346, 367, 381, 388], 8: [96, 185, 301, 318, 325, 406], 9: [18, 32, 48, 61, 79, 152, 242, 280, 284, 298, 340, 453], 10: [52, 53, 54, 63, 71, 72, 75, 80, 85, 176, 178, 272, 282, 283, 286, 294, 384, 399], 11: [73, 74, 82, 84, 90, 177, 190, 200, 202, 207, 210, 300, 314, 405], 12: [30, 35, 37, 39, 45, 49, 57, 59, 60, 69, 76, 140, 141, 142, 147, 149, 150, 155, 169, 174, 175, 251, 253, 261, 264, 268, 285, 290, 304, 352, 356, 363, 379, 382, 383, 391, 404], 13: [81, 83, 86, 87, 91, 92, 100, 182, 189, 191, 193, 196, 199, 201, 203, 211, 213, 216, 219, 307, 309, 310, 312, 395, 398, 403, 408, 409, 412, 418, 419, 420, 423, 428], 14: [0, 93, 97, 105, 109, 118, 183, 192, 217, 303, 305, 397, 434, 437, 445, 446], 15: [43, 46, 51, 55, 58, 148, 153, 162, 164, 265, 273, 275, 299, 372, 385, 401], 16: [9, 88, 101, 195, 208, 227, 234, 292, 321, 335, 417, 421, 436, 440, 457]}

    result_integration(all_result, result_cluster, track_list)  # 把聚类结果进行整合
    print("聚类结果输出：", all_result)
    print(len(all_result))
    # 聚类结果分析:
    Similarity_analysis(data, all_result, Draw=False)
    # Pre_Calculate = True  # 如果已经计算出最小连通图，TRUE直接对每个连通图进行聚类，False则需要提前计算
    # if not Pre_Calculate:
    #     Minimum_connected_graph = ISOMAP.Dijkstra(similar_matrix[:, :, 0])
    # for i, Minimum_connected in enumerate(Minimum_connected_graph):
    #     if len(Minimum_connected) < 5:  # 记为噪声
    #         print("{}为噪声".format(Minimum_connected))
    #         continue
    #     # 提取最小连接图对应的tr_id, similar_matrix相似度矩阵
    #     partial_tr_list, partial_similar_matrix = ISOMAP.Extract_Corresponding_Value(tr_list, similar_matrix,
    #                                                                                  Minimum_connected)
    #     # 对数据进行降维
    #     if os.path.exists(os.path.join(OUTPUT_PATH, '{}_point_matrix.npy'.format(Minimum_connected[0]))):
    #         # 直接读取文件，不需要重新计算全距离矩阵，且距离映射，因为解有很多个结果可能会有不同：
    #         point_matrix = np.load(os.path.join(OUTPUT_PATH, '{}_point_matrix.npy'.format(Minimum_connected[0])))
    #     else:
    # 计算邻近轨迹的距离，然后根据最短路径算法计算得到dist(x,y)，获得全局距离矩形，调用MDS算法获得降维数据
    #         isomap = ISOMAP(n_components=1)
    #         x = isomap.fit(partial_similar_matrix[:, :, 0]).reshape((len(partial_tr_list), 1))  # [变化趋势]
    #         y = isomap.fit(partial_similar_matrix[:, :, 1]).reshape((len(partial_tr_list), 1))  # [初尾位置]
    #         z = isomap.fit(partial_similar_matrix[:, :, 2]).reshape((len(partial_tr_list), 1))  # [时间戳相似度]
    #         point_matrix = np.concatenate([x, y, z], axis=1)
    #         # 数据归一化（方便归一化）
    #         for j in range(point_matrix.shape[-1]):
    #             point_matrix[:, j] = (point_matrix[:, j] - min(point_matrix[:, j])) / (max(point_matrix[:, j]) -
    #                                                                                    min(point_matrix[:, j]))
    #         np.save(os.path.join(OUTPUT_PATH, '{}_point_matrix.npy'.format(Minimum_connected[0])), point_matrix)
    #         print('{}文件存储完毕！'.format(os.path.join(OUTPUT_PATH, '{}_point_matrix.npy'.format(
    #         Minimum_connected[0]))))
    # # import umap
    # # umap = umap.UMAP(n_components=1)  # 只需要计算KNN轨迹间的距离，但是距离会被拉伸
    # # point_matrix = np.concatenate([umap.fit_transform(similar_matrix[:, :, 0]),
    # #                                umap.fit_transform(similar_matrix[:, :, 1]),
    # #                                umap.fit_transform(similar_matrix[:, :, 2])], axis=1)
    # # timespan = []
    # # from Similarity_Measurement_Methodology import time_change
    # # for i, tr in enumerate(tr_list):
    # #     timespan.append(time_change(tr.TimeSpan[0]), time_change(tr.TimeSpan[1]))
    # # point_matrix = np.concatenate([point_matrix, np.array(timespan)], axis=1)
    # # CC(point_matrix)  # 用R语言CC算法确定K值
    # # K = 10
    # # optics_parameters(similar_matrix, 0, draw=False)
    # # for i in range(5):
    #     # eps_list = KANN(similar_matrix)[0:10]
    #     print("轨迹聚类-----------------------------------------------------------------------------------\n")
    #     model = ['DPC']  # 'AP', 'KMeans', 'SC', 'OPTICS', 'DPC', 'FCM'
    #     result_cluster = Cluster_model(model, point_matrix, K=4, similar_matrix=partial_similar_matrix, draw=True)
    #     result_cluster = Cluster_model(model, point_matrix, K=4, similar_matrix=None, draw=True)
    #     result_cluster = consolidation(label_, center)
    #     Result_integration(all_result, result_cluster, partial_tr_list)  # 把聚类结果进行整合
    # print("聚类结果输出：", all_result)
    # print(len(all_result))
    # 结果整合后可视化
    # mv = ModelVisualization(track_dict=data, result_dict=all_result)
    # mv.track_point_visualization(result_cluster, point_matrix,
    #                              filepath=os.path.join(OUTPUT_PATH, 'all_result_dic.html'))  # 轨迹点聚类3D散点图
    # mv.pos_heatmap()  # 轨迹的经纬度热点图
    # mv.xyt_line(Save=True, Show=False)
    # mv.corr_surface(Save=True, Show=False)  # 轨迹聚类的协方差3D曲面图，和每个聚类的平均协方差的平面图


    # if Data_visualization:
    #     # point_matrix 数据可视化聚类效果x:变化趋势, y:初尾位置, z:时间戳相似度, result{cluster_id:[tr_id...],...}
    #     from Data_Analysis.Data_Visualization import Point_Clustering_Visualization, KED
    #     Point_Clustering_Visualization(result_cluster, point_matrix)
    #     # 所有轨迹聚类的效果图
    #     Track_Clustering_Visualization(data, all_result)  # 每个簇的各个维度上的聚类结果
    #     # 把数据点的经纬度展示出
    #     KED(data, all_result)  # 经纬度热点图
    #     Trace_plot(data, all_result)  # 经纬度平面图
    #     # 每个簇corr 3D曲面图
    #     CorrSurface(data, all_result)
    #     # 数据降维（ISOMAP）2维 ，展示3维图中（z轴时间序列）——代表轨迹 还是所有轨迹呢？
    #     ND_plot(data, all_result)

    DATA_SAVE = False  # 是否做数据存储
    if DATA_SAVE:
        # 数据保存
        save_matrix = []
        for i in data.keys():
            for j in data[i].point_list:
                save_matrix.append(j.origin_array() + [data[i].ClusterId])
        df = pd.DataFrame(data=save_matrix, columns=['track', 'time', 'amplitude', 'longitude', 'latitude',
                                                     'speed_average', 'effective_radius', 'velocity', 'angle',
                                                     'ClusterId'])
        df.to_csv(os.path.join(OUTPUT_PATH, 'result.csv'), index=False)
        print('{} file saved success.'.format(os.path.join(OUTPUT_PATH, 'result.csv')))
