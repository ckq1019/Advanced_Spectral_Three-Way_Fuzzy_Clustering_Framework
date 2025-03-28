import os.path
import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import time
# import sys
# sys.path.append(os.path.join(os.getcwd(),  ".."))

Track_id = 1  # 轨迹id


def make_trajectory(traj_array, traj_list, labels, Time=None):
    '''
    制作轨迹，计算其（id, t, x, y , v, a, label）
    :param traj_list: 轨迹数据存放列表
    :param traj_array: 轨迹数据集
    :param labels:该轨迹数据集的标签
    :param Time: 如果为None的话就没有时间属性t，否则的话加入为中间时间段
    :return:
    '''
    global Track_id  # 全局变量
    for row in range(traj_array.shape[0]):  # 遍历每条轨迹，每一行就是一条轨迹
        xy = traj_array[row].reshape((1, -1))
        xy = xy.reshape((2, -1))
        ran_seed = random.randint(-3, 3)  # 随机数
        for i in range(xy.shape[1]):
            x = xy[0][i]
            y = xy[1][i]
            if i == xy.shape[1]-1:
                v = a = 0  # 轨迹中的最后一点
            else:
                v, a = cal_va(x, y, xy[0][i+1], xy[1][i+1])
            if Time is not None:
                t = Time + i + ran_seed
                traj_list.append([Track_id, t, x, y, v, a, labels])
            else:
                traj_list.append([Track_id, x, y, v, a, labels])
        Track_id += 1


def cal_va(x1, y1, x2, y2):
    v = math.sqrt(math.pow(y2 - y1, 2) + math.pow(x2 - x1, 2))
    if x1 == x2:
        angle = math.pi / 2 if (y2 - y1) > 0 else math.pi + math.pi / 2
    else:
        if (y2 - y1) >= 0 and (x2 - x1) > 0:
            angle = np.arctan((y2 - y1) / (x2 - x1))
        elif (y2 - y1) >= 0 and (x2 - x1) < 0:
            angle = math.pi + np.arctan((y2 - y1) / (x2 - x1))
        elif (y2 - y1) < 0 and (x2 - x1) < 0:
            angle = math.pi + np.arctan((y2 - y1) / (x2 - x1))
        elif (y2 - y1) < 0 and (x2 - x1) > 0:
            angle = np.arctan((y2 - y1) / (x2 - x1)) + math.pi * 2
    a = angle * 180 / math.pi
    return v, a


def create_data():
    # 分为8个簇，长度各不相同，速度不同，时间不同
    # 1.首先生成200（其中一半时间不同），200（v * 2），200（时间段不同），100， 100
    # 时间段1： 10:180；时间段2：190-360。时间段1进行细分10-80，110-180，     时间段2都是用相似的开始时间
    # [TS,anom] = create_ts2(200, 1, 1, 1, 15, 0.7);时间段1
    # [TS, anom] = create_ts2(200, 1, 1, 1, 20, 0.7);时间段2
    # [TS, anom] = create_ts2(200, 1, 1, 1, 20, 0.6);时间段1+时间段2
    # [TS,anom] = create_ts2(100, 1, 1, 1, 30, 0.7);时间段1
    # [TS,anom] = create_ts2(100, 1, 1, 1, 25, 0.7);时间段2
    # csvwrite("D:\1.csv",TS)
    file_path = os.path.join(os.getcwd(), "data")
    cluster1 = pd.read_csv(os.path.join(file_path, "1.csv"), header=None)
    cluster1 = np.array(cluster1)
    traj_list = []
    make_trajectory(cluster1[:50], traj_list, 0, Time=85-20)
    make_trajectory(cluster1[50:100], traj_list, 1, Time=85+20)

    cluster2 = pd.read_csv(os.path.join(file_path, "2.csv"), header=None)
    cluster2 = np.array(cluster2)
    make_trajectory(cluster2[:50], traj_list, 2, Time=275 - 10)
    # v * 2
    make_trajectory(cluster2[50:100, np.where(np.arange(cluster2.shape[1]) % 2 == 0)[0]], traj_list, 3, Time=275 - 5)

    cluster3 = pd.read_csv(os.path.join(file_path, "3.csv"), header=None)
    cluster3 = np.array(cluster3)
    make_trajectory(cluster3[:50], traj_list, 4, Time=275 - 10)
    make_trajectory(cluster3[50:100, ], traj_list, 5, Time=85 - 10)

    cluster4 = pd.read_csv(os.path.join(file_path, "4.csv"), header=None)
    cluster4 = np.array(cluster4)
    make_trajectory(cluster4[:50], traj_list, 6, Time=85 - 15)

    cluster5 = pd.read_csv(os.path.join(file_path, "5.csv"), header=None)
    cluster5 = np.array(cluster5)
    make_trajectory(cluster5[:50], traj_list, 7, Time=275-12)

    noise = pd.read_csv(os.path.join(file_path, "noise.csv"), header=None)
    for row in noise.index:
        noise_array = np.array(noise.iloc[row], dtype=np.float64)
        noise_array = noise_array[~np.isnan(noise_array)].reshape((1, -1))
        make_trajectory(noise_array, traj_list, -1, Time=random.randint(1, int(366-len(noise[row])/2)))

    df = pd.DataFrame(traj_list, index=None, columns=["track_id", "time", "x", "y", "v", "a", "label"])
    df.to_csv(os.path.join(file_path, "Dataset2.csv"), index=None)

    # 不加时间的一组——考察除了时间外的相似度度量
    # [TS,anom] = create_ts2(200, 1, 1, 1, 15, 0.7);方向不同
    # [TS, anom] = create_ts2(200, 1, 1, 1, 20, 0.7);速度不同
    # [TS, anom] = create_ts2(200, 1, 1, 1, 20, 0.6);前100
    # [TS,anom] = create_ts2(100, 1, 1, 1, 30, 0.7);
    # [TS,anom] = create_ts2(100, 1, 1, 1, 25, 0.7);
    # traj_list = []
    # global Track_id
    # Track_id = 1
    # make_trajectory(cluster1[:50], traj_list, 0, Time=None)
    # reverse = cluster1[50:100]
    # reverse[:, :15] = reverse[:, np.arange(14, -1, -1)]
    # reverse[:, 15:] = reverse[:, np.arange(29, 14, -1)]
    # make_trajectory(reverse, traj_list, 1, Time=None)  # 方向相反
    #
    # make_trajectory(cluster2[:50], traj_list, 2, Time=None)
    # make_trajectory(cluster2[50:100, np.where(np.arange(cluster2.shape[1]) % 2 == 0)[0]], traj_list, 3, Time=None)
    #
    # make_trajectory(cluster3[:50], traj_list, 4, Time=None)
    #
    # make_trajectory(cluster4[:50], traj_list, 5, Time=None)
    #
    # make_trajectory(cluster5[:50], traj_list, 6, Time=None)
    #
    # noise = pd.read_csv(os.path.join(file_path, "noise.csv"), header=None)
    # for row in noise.index:
    #     noise_array = np.array(noise.iloc[row], dtype=np.float64)
    #     noise_array = noise_array[~np.isnan(noise_array)].reshape((1, -1))
    #     make_trajectory(noise_array, traj_list, -1, Time=None)
    #
    # df = pd.DataFrame(traj_list, index=None, columns=["track_id", "x", "y", "v", "a", "label"])
    # df.to_csv(os.path.join(file_path, "Dataset1.csv"), index=None)


def ACC(true, predict):
    '''
    精确度
    :param true:
    :param predict:
    :return:
    '''
    true = np.array(true)
    predict = np.array(predict)
    accuracy = []  # 准确率
    precision = []  # 精确率
    recall = []  # 召回率
    track_num = len(true)  # 轨迹数量
    for track_index in range(track_num):  # 遍历每个轨迹
        true_value = np.zeros(track_num)
        predict_value = np.zeros(track_num)
        true_value[np.where(true == true[track_index])[0]] = 1
        predict_value[np.where(predict == predict[track_index])[0]] = 1
        TP = len(np.where((true_value == 1) & (predict_value == 1))[0])
        FP = len(np.where((true_value == 0) & (predict_value == 1))[0])
        FN = len(np.where((true_value == 1) & (predict_value == 0))[0])
        TN = len(np.where((true_value == 0) & (predict_value == 0))[0])
        accuracy.append((TP + TN) / (TP + FN + TN + FP))
        precision.append(TP / (TP + FP))
        recall.append(TP/(TP+FN))
    print("Accuracy: ", np.mean(accuracy))
    print("Precision: ", np.mean(precision))
    print("Recall: ", np.mean(recall))


def Purity(true, predict):
    '''
    纯度
    :param true:正确label
    :param predict:预测label
    :return:
    '''
    true = np.array(true)
    predict = np.array(predict)
    purity = 0
    samples_num = len(true)  # 样本数量
    true_labels = list(set(true))  # 簇标签
    predict_labels = list(set(predict))
    for tlabel in true_labels:  # 遍历每个簇
        if tlabel == -1:
            purity += len(np.where((predict == -1) & (true == -1))[0])
            continue
        pure = []
        for plabel in predict_labels:  # 遍历每个预测标签
            pure.append(len(np.where((predict == plabel) & (true == tlabel))[0]))
        purity += max(pure)
    print("Purity: ", purity / samples_num)


def LWDTW(tracka_point, trackb_point, Weight=0.5, Time=None):
    if Time is not None:
        start_time = time.time()  # 开始时间
    Distance = []  # 把各特征的计算距离放入W=1
    for Character in range(tracka_point.shape[1]):
        # 进行轨迹分割
        # from Trajectory_Segmentation.Trajectory_Segmentation_Methodology import myMethodology
        # key_a = myMethodology(tracka_point[:, Character])  # 轨迹a的得到的关键点[[0, x[0]], [index, x[index]], ...]
        # key_b = myMethodology(trackb_point[:, Character])
        key_a = np.concatenate([np.arange(tracka_point.shape[0]).reshape((-1, 1)),
                                tracka_point[:, Character].reshape((-1, 1))], axis=1)
        key_b = np.concatenate([np.arange(trackb_point.shape[0]).reshape((-1, 1)),
                                trackb_point[:, Character].reshape((-1, 1))], axis=1)

        DTW = {}  # 初始化DTW矩阵
        w = abs(len(key_a) - len(key_b)) + 1
        for i in range(len(key_a) - 1):
            DTW[(i, -1)] = float('inf')
        for i in range(len(key_b) - 1):
            DTW[(-1, i)] = float('inf')
        for i in range(len(key_a) - 1):
            DTW[(i, i + w)] = float('inf')
            DTW[(i, i - w - 1)] = float('inf')
        DTW[(-1, -1)] = 0
        weight_DTW = DTW.copy()  # 加权DTW
        # DTW计算
        for i in range(len(key_a) - 1):
            for j in range(max(0, i - w), min(len(key_b) - 1, i + w)):
                # 两个线段之间的距离（斜率之差*0.4+初末点之差*0.5*0.6）
                dist = (abs(key_a[i, 1] - key_b[j, 1]) + abs(key_a[i + 1, 1] - key_b[j + 1, 1])) * 0.5 * (1-Weight) + \
                       abs((key_a[i + 1, 1] - key_a[i, 1]) / (key_a[i + 1, 0] - key_a[i, 0]) -
                           (key_b[j + 1, 1] - key_b[j, 1]) / (key_b[j + 1, 0] - key_b[j, 0])) * Weight
                Mininum = min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])  # 最小值定位
                if Mininum == DTW[(i - 1, j)]:
                    MiniPos = (i - 1, j)
                elif Mininum == DTW[(i, j - 1)]:
                    MiniPos = (i, j - 1)
                else:
                    MiniPos = (i - 1, j - 1)
                DTW[(i, j)] = dist + Mininum
                weight_DTW[(i, j)] = dist * ((key_a[i + 1, 0] - key_a[i, 0]) / (key_a[-1, 0] - key_a[0, 0]) +
                                             (key_b[j + 1, 0] - key_b[j, 0]) / (key_b[-1, 0] - key_b[0, 0])) + \
                                     weight_DTW[MiniPos]
        Distance.append(weight_DTW[(len(key_a) - 2, len(key_b) - 2)])
    print("变化趋势相似度：", np.mean(Distance))
    if Time is not None:
        end_time = time.time()
        Time[0] += (end_time - start_time)
    return np.mean(Distance)


def DTW(tracka_point, trackb_point, Time=None):
    if Time is not None:
        start_time = time.time()
    n, m = tracka_point.shape[0], trackb_point.shape[0]

    def Distance(i, j):
        dist = np.linalg.norm(tracka_point[i]-trackb_point[j], 2)
        return dist

    dtw_matrix = np.full((n, m), np.inf)
    dtw_matrix[0, 0] = Distance(0, 0)
    for i in range(0, n):
        for j in range(1, m):
                dtw_matrix[i, j] = Distance(i, j) + min(dtw_matrix[i - 1, j], dtw_matrix[i - 1, j - 1],
                                                        dtw_matrix[i, j - 1])
    print("DTW距离相似度：", dtw_matrix[n-1, m-1])
    if Time is not None:
        end_time = time.time()
        Time[1] += (end_time - start_time)
    return dtw_matrix[n-1, m-1]


def Position_Similarity(tracka_point, trackb_point):
    L_a = [list(tracka_point[0][:2]), list(tracka_point[-1][:2])]
    L_b = [list(trackb_point[0][:2]), list(trackb_point[-1][:2])]

    def p_line(s, e, p):
        '''
        求点p到由se组成的直线的垂直距离
        :param s: s[x1,y1]
        :param e: e[x2,y2]
        :param p: p[x3,y3]
        :return:
        '''
        if math.sqrt(pow((s[1] - e[1]), 2) + pow(s[0] - e[0], 2)) == 0:
            return 0
        else:
            return np.float64(
                math.fabs((s[1] - e[1]) * p[0] + (e[0] - s[0]) * p[1] + s[0] * e[1] - s[1] * e[0]) / math.sqrt(
                    (s[1] - e[1]) ** 2 + (s[0] - e[0]) ** 2))

    def dist(a, b):
        return math.sqrt(math.pow(b[1] - a[1], 2) + math.pow(b[0] - a[0], 2))

    a = p_line(L_b[0], L_b[1], L_a[0])
    b = p_line(L_b[0], L_b[1], L_a[1])

    def d_prp():
        '''
        计算两线段之间的垂直距离
        :param L_a:
        :param L_b:
        :return:
        '''
        if a == 0 and b == 0:
            return 0
        return a ** 2 + b ** 2

    def d_prl():
        '''
        计算两线段之间的平行距离
        :param L_a:
        :param L_b:
        :return:
        '''
        if L_b == L_a:
            return 0
        s_prl = math.sqrt(dist(L_a[0], L_b[0]) ** 2 - a ** 2)
        e_prl = math.sqrt(dist(L_a[1], L_b[1]) ** 2 - b ** 2)
        return s_prl ** 2 + e_prl ** 2

    def d_angle():
        '''
        计算两线段间的角度距离
        :param L_a:
        :param L_b:
        :return:
        '''
        if L_a == L_b:
            return 0
        return (a - b) ** 2

    pos_similar = math.sqrt(d_prp() + d_prl() + d_angle())
    print("轨迹初末点空间相似度：", pos_similar)
    return pos_similar


def HD(tracka_point, trackb_point, Time = None):
    if Time is not None:
        start_time = time.time()
    n, m = tracka_point.shape[0], trackb_point.shape[0]
    dist_matrix = np.zeros((n + 1, m + 1))
    from Similarity.Similarity_Measurement_Methodology import Multifactor_distance
    for i in range(n):
        for j in range(m):
            dist_matrix[i][j] = Multifactor_distance(tracka_point[i], trackb_point[j], cal_type='Point')
            if dist_matrix[n][j] == 0:
                dist_matrix[n][j] = dist_matrix[i][j]
            elif dist_matrix[n][j] > dist_matrix[i][j]:
                dist_matrix[n][j] = dist_matrix[i][j]
        dist_matrix[i][j + 1] = np.min(dist_matrix[i][:-1])
    dist_matrix[i + 1][j] = np.max(dist_matrix[i + 1][:-1])
    print('MFHD距离相似度： ', max(dist_matrix[:, -1]))
    if Time is not None:
        end_time = time.time()
        Time[2] += (end_time - start_time)
    return max(dist_matrix[:, -1])


def Timestamp_Similarity(time_a, time_b):
    tra_start = int(time_a[0])
    tra_end = int(time_a[-1])
    trb_start = int(time_b[0])
    trb_end = int(time_b[-1])

    start_time = min([tra_start, trb_start])  # 两个轨迹最小的轨迹点
    end_time = max([tra_end, trb_end])  # 两个轨迹比较大的时间戳
    time_similar = 1 + (max(tra_start, trb_start) - min(tra_end, trb_end)) / (end_time - start_time)
    print("时间相似度", time_similar)
    return time_similar


def convert_to_label(membership, noise_threshold, Labels=None):
    '''
    根据隶属度转换成标签
    :param membership:簇隶属度
    :param noise_threshold:噪声阈值
    :param Labels:设置标签数字
    :return:
    '''
    labels = np.full(membership.shape[0], -1)
    for row in range(membership.shape[0]):
        if len(np.where(membership[row] >= noise_threshold)[0]) == 0:  # 噪声
            continue
        if len(np.where(membership[row] != 0)[0]) == 1:
            labels[row] = np.where(membership[row] != 0)[0][0]
            continue
        labels[row] = np.argmax(membership[row])
    if Labels is None:
        return labels
    else:
        labels[labels != -1] = labels[labels != -1] + min(Labels)
        return labels


def Spectral_Clustering_FCM(n_clusters, Similar_matrix, Weight):
    W = np.zeros((Similar_matrix.shape[0], Similar_matrix.shape[0]), dtype=np.float64)
    for i in range(Similar_matrix.shape[0]):
        for j in range(i, Similar_matrix.shape[0]):
            W[j][i] = W[i][j] = Weight[i] * Weight[j]

    # 计算邻接矩阵A
    def AKNN(S, sigma=1.0):
        N = S.shape[0]
        A = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                A[i][j] = np.exp(-S[i][j] ** 2 / 2 / sigma / sigma) * W[i, j]
                # 高斯核函数:随着两个向量的距离增大，高斯核函数单调递减
                # np.exp=e^x
                A[j][i] = A[i][j]  # mutually
        return A

    Adjacent = AKNN(Similar_matrix, sigma=1.3)

    def calLaplacianMatrix(adjacentMatrix):
        degreeMatrix = np.sum(adjacentMatrix, axis=1)
        laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix
        sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** 0.5))
        return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)

    Laplacian = calLaplacianMatrix(Adjacent)

    # 特征值分解，计算方形矩阵Laplacian的特征值和特征向量, 得到特征向量Hnn,

    x, V = np.linalg.eig(Laplacian)
    x = zip(x, range(len(x)))
    x = sorted(x, key=lambda x: x[0])
    # 按列表的第一个元素(特征值)升序排列
    H = np.vstack([V[:, i] for (v, i) in x[:n_clusters]]).T

    from Cluster.FCM import FCM
    return FCM(H, n_clusters, Membership=True)


class MV_STFC:
    def __init__(self, num_track, file_path=None):
        self.num_track = num_track  # 轨迹数量
        self.run_time = 0  # 运行时间
        self.Trajectory_dictionary = {}  # 轨迹字典{track_index: track_id}
        self.similar_mat = np.zeros((num_track, num_track, 3))  # 相似度矩阵
        self.file_path = file_path  # 文件位置
        self.predict_labels = np.full(num_track, -1)  # 预测标签
        self.predict_membership = None  # 预测的隶属度

    def Cluster(self, n_clusters, track_df, read_file=False):
        if read_file:
            self.read_file(self.file_path)
        else:
            track_id = list(track_df["track_id"].unique())
            start_time = time.time()
            # 相似度计算
            for i, track_a in enumerate(track_id):
                self.Trajectory_dictionary[i] = track_a  # 轨迹id
                for j, track_b in enumerate(track_id):
                    if track_a == track_b or self.similar_mat[i][j][0] != 0:
                        continue
                    tracka_point = np.array(track_df[track_df["track_id"] == track_a][["x", "y", "v", "a"]],
                                            dtype=np.float64)
                    time_a = np.array(track_df[track_df["track_id"] == track_a][["time"]])
                    trackb_point = np.array(track_df[track_df["track_id"] == track_b][["x", "y", "v", "a"]],
                                            dtype=np.float64)
                    time_b = np.array(track_df[track_df["track_id"] == track_b][["time"]])
                    self.similar_mat[i][j][0] = self.similar_mat[j][i][0] = LWDTW(tracka_point, trackb_point,
                                                                                  Weight=0.5)
                    self.similar_mat[i][j][1] = self.similar_mat[j][i][1] = Position_Similarity(tracka_point,
                                                                                                trackb_point)
                    self.similar_mat[i][j][2] = self.similar_mat[j][i][2] = Timestamp_Similarity(time_a, time_b)
            # 归一化
            for j in range(3):
                self.similar_mat[:, :, j] = (self.similar_mat[:, :, j] - min(self.similar_mat[:, :, j].flat)) / (
                    max(self.similar_mat[:, :, j].flat) - min(self.similar_mat[:, :, j].flat))
            # 一致性相似度
            from Cluster.MultiView_Spectral_Clustering import Consensus_similarity
            consensus_similarity = Consensus_similarity(self.similar_mat)
            # 聚类
            from Cluster.SC import mySpectral_Clustering
            weight = np.ones(self.similar_mat.shape[0])
            self.predict_membership = mySpectral_Clustering(n_clusters, Similar_matrix=consensus_similarity, Weight=weight)
            end_time = time.time()
            self.predict_labels = convert_to_label(self.predict_membership, 0.2)
            self.run_time += (end_time - start_time)

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

    def Save_results(self, filepath):  # 结果保存
        np.save(os.path.join(filepath, "MVSTFC_labels.npy"), self.predict_labels)
        np.save(os.path.join(filepath, "MVSTFC_membership.npy"), self.predict_membership)
        np.save(os.path.join(filepath, "MVSTFC_similar_mat.npy"), self.similar_mat)
        np.save(os.path.join(filepath, "MVSTFC_Trajectory_dictionary.npy"), self.Trajectory_dictionary)
        np.save(os.path.join(self.file_path, "MVSTFC_run_time.npy"), self.run_time)
        print("Results saved successfully.")

    def read_file(self, filepath):  # 直接读取数据结果
        self.predict_labels = np.load(os.path.join(filepath, "MVSTFC_labels.npy"))
        self.predict_membership = np.load(os.path.join(filepath, "MVSTFC_membership.npy"))
        self.similar_mat = np.load(os.path.join(filepath, "MVSTFC_similar_mat.npy"))
        self.Trajectory_dictionary = np.load(os.path.join(filepath, "MVSTFC_Trajectory_dictionary.npy"),
                                             allow_pickle=True).item()
        self.run_time = np.float64(np.load(os.path.join(self.file_path, "MVSTFC_run_time.npy")))
        print("Results read successfully.")


class Model:  # 聚类模型
    def __init__(self, num_track, file_path=None):
        self.run_time = 0  # 运行时间
        self.similar_mat = np.zeros((num_track, num_track))  # 相似度矩阵
        self.file_path = file_path  # 文件位置
        self.predict_labels = np.full(num_track, -1)  # 预测标签

    def Evaluation(self, track_df, trid, true_labels):
        clusters = list(set(self.predict_labels))  # 簇标签
        time_std = np.zeros((len(clusters), 2))  # 记录每个簇的时间戳初末的方差
        for i, cluster_id in enumerate(clusters):  # 遍历每个簇
            timespan = []  # 存储每个簇的时间跨度[[start0, end0],[start1, end1]]
            for track_idx in np.where(self.predict_labels == cluster_id)[0]:
                track_id = trid[track_idx]
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
                track_id = trid[track_idx]
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
                track_id = trid[track_idx]
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

        from sklearn.metrics import normalized_mutual_info_score, silhouette_score
        print("NMI:", normalized_mutual_info_score(true_labels, self.predict_labels))
        print("Run time:", self.run_time)

    def keep_file(self):  # 存档
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
        np.save(os.path.join(filepath, "result.npy"), self.predict_labels)
        print("Result saved successfully.")


if __name__ == '__main__':
    # 制作数据集
    # create_data()
    file_path = os.path.join(os.getcwd(), "data")

    df = pd.read_csv(os.path.join(file_path, "Dataset1.csv"))  # 读取数据
    # 数据归一化
    for col in ["x", "y", "v", "a"]:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    track_id = list(df["track_id"].unique())  # 所有的轨迹id
    print(len(track_id))
    # true_labels = []  # 真实标签
    # # 1.比较不加时间要素：计算相似度度量比较:LWDTW, DTW,HD
    # time_list = [0, 0, 0]
    # similar_mat = np.zeros((len(track_id), len(track_id), 3))
    # for i, a_id in enumerate(track_id):
    #     track_a = df[df["track_id"] == a_id]
    #     true_labels.append(int(track_a["label"].unique()))  # 真实轨迹标签
    #     track_a = track_a[["x", "y", "v", "a"]]
    #     for j, b_id in enumerate(track_id):
    #         if i == j or similar_mat[i][j][0] != 0:
    #             continue
    #         track_b = df[df["track_id"] == b_id]
    #         track_b = track_b[["x", "y", "v", "a"]]
    #         similar_mat[i, j, 0] = similar_mat[j, i, 0] = LWDTW(
    #             np.array(track_a, dtype=np.float64), np.array(track_b, dtype=np.float64), 0.5, time_list)
    #         similar_mat[i, j, 1] = similar_mat[j, i, 1] = DTW(
    #             np.array(track_a, dtype=np.float64), np.array(track_b, dtype=np.float64), time_list)
    #         similar_mat[i, j, 2] = similar_mat[j, i, 2] = HD(
    #             np.array(track_a, dtype=np.float64), np.array(track_b, dtype=np.float64), time_list)
    # file_path = os.path.join(os.getcwd(), "output")
    # np.save(os.path.join(file_path, "similar_mat.npy"), similar_mat)
    # print("{} file saved sucessful.".format(file_path))
    # np.save(os.path.join(file_path, "true_labels.npy"), np.array(true_labels))
    # print("{} file saved sucessful.".format(file_path))
    # print("time0: {} \ntime1: {} \ntime2: {} \n".format(time_list[0], time_list[1], time_list[2]))

    file_path = os.path.join(os.getcwd(), "output")
    similar_mat = np.load(os.path.join(file_path, "similar_mat.npy"))
    true_labels = list(np.load(os.path.join(file_path, "true_labels.npy")))
    for method_index in range(similar_mat.shape[-1]):
        from Cluster.SC import Spectral_Clustering
        predict_labels = Spectral_Clustering(7, n_neighborhood=5, Similar_matrix=similar_mat[:, :, method_index])
        from sklearn.cluster import AffinityPropagation, OPTICS, SpectralClustering
        predict_labels = AffinityPropagation(affinity="precomputed").fit_predict(similar_mat[:, :, method_index])
        predict_labels = OPTICS(metric="precomputed", min_samples=40).fit_predict(similar_mat[:, :, method_index])
        predict_labels = SpectralClustering(n_clusters=7, affinity="precomputed").fit_predict(similar_mat[:, :, method_index])
        # 计算精度，召回率
        from sklearn.metrics import normalized_mutual_info_score, accuracy_score, adjusted_rand_score, recall_score, \
            f1_score
        # print("ACC:", accuracy_score(true_labels, predict_labels))
        print("NMI:", normalized_mutual_info_score(true_labels, predict_labels))
        # print("F-Score:", f1_score(true_labels, predict_labels, average="micro"))
        print("ARI:", adjusted_rand_score(true_labels, predict_labels))
        # print("Recall:", recall_score(true_labels, predict_labels, average="micro"))
        print("-" * 40)

    # 2.1 加上时间要素，相似度比较
    # similar_mat = np.zeros((len(track_id), len(track_id), 3))
    # for a_index in range(len(track_id)):
    #     track_a = df[df["track_id"] == track_id[a_index]]
    #     true_labels.append(int(track_a["label"].unique()))  # 真实轨迹标签
    #     time_a = track_a["time"]
    #     track_a = track_a[["x", "y", "v", "a"]]
    #     for b_index in range(a_index, len(track_id)):
    #         if a_index == b_index:
    #             continue
    #         track_b = df[df["track_id"] == track_id[b_index]]
    #         time_b = track_b["time"]
    #         track_b = track_b[["x", "y", "v", "a"]]
    #         similar_mat[a_index, b_index, 0] = similar_mat[b_index, a_index, 0] = LWDTW(
    #             np.array(track_a, dtype=np.float64), np.array(track_b, dtype=np.float64), Weight=0.5)
    #         similar_mat[a_index, b_index, 1] = similar_mat[b_index, a_index, 1] = Position_Similarity(
    #             np.array(track_a, dtype=np.float64), np.array(track_b, dtype=np.float64))
    #         similar_mat[a_index, b_index, 2] = similar_mat[b_index, a_index, 2] = Timestamp_Similarity(
    #             np.array(time_a, dtype=np.float64), np.array(time_b, dtype=np.float64))
    # file_path = os.path.join(os.getcwd(), "output")
    # np.save(os.path.join(file_path, "add_time_similar_mat.npy"), similar_mat)
    # print("{} file saved sucessful.".format(file_path))
    # np.save(os.path.join(file_path, "add_time_true_labels.npy"), np.array(true_labels))
    # print("{} file saved sucessful.".format(file_path))

    # similar_mat = np.load(os.path.join(file_path, "add_time_similar_mat.npy"))
    # true_labels = list(np.load(os.path.join(file_path, "add_time_true_labels.npy")))
    # weight = np.ones(similar_mat.shape[0])
    # # 归一化
    # for i in range(similar_mat.shape[-1]):
    #     similar_mat[:, :, i] = (similar_mat[:, :, i] - min(similar_mat[:, :, i].flat)) / (
    #             max(similar_mat[:, :, i].flat) - min(similar_mat[:, :, i].flat))
    # for method_index in [[0, 1, 2], [0, 1], [0, 2], [0], 0]:
    #     if type(method_index) == list and len(method_index) > 1:
    #         from Cluster.MultiView_Spectral_Clustering import Consensus_similarity
    #         consensus_similarity = Consensus_similarity(similar_mat[:, :, method_index])
    #         from Cluster.SC import mySpectral_Clustering
    #         predict_membership = mySpectral_Clustering(8, Similar_matrix=consensus_similarity, Weight=weight)
    #     elif type(method_index) == list and len(method_index) == 1:
    #         from Cluster.SC import mySpectral_Clustering
    #         predict_membership = mySpectral_Clustering(6, Similar_matrix=similar_mat[:, :, method_index[0]],
    #                                                    Weight=weight)
    #     else:
    #         similarity_mat = similar_mat[:, :, 0] ** 2 + similar_mat[:, :, 1] ** 2 + similar_mat[:, :, 2] ** 2
    #         predict_membership = mySpectral_Clustering(6, Similar_matrix=similarity_mat, Weight=weight)
    #     predict_labels = convert_to_label(predict_membership, noise_threshold=0.2)
    #     # 计算精度，召回率
    #     from sklearn.metrics import normalized_mutual_info_score, accuracy_score, adjusted_rand_score, recall_score, \
    #         f1_score
    #     print("ACC:", accuracy_score(true_labels, predict_labels))
    #     print("NMI:", normalized_mutual_info_score(true_labels, predict_labels))
    #     print("F-Score:", f1_score(true_labels, predict_labels, average="macro"))
    #     print("ARI:", adjusted_rand_score(true_labels, predict_labels))
    #     print("Recall:", recall_score(true_labels, predict_labels, average="macro"))
    #     print("-" * 40)
    #
    # # 2.2 其他聚类（算法最后用FCM , 聚类算法改成OPTICS, CS, AP）
    #
    # from Cluster.MultiView_Spectral_Clustering import Consensus_similarity
    # consensus_similarity = Consensus_similarity(similar_mat)
    # for method_index in ["SFCM", "OPTICS", "SC", "AP"]:
    #     if method_index == "SFCM":
    #         predict_membership = Spectral_Clustering_FCM(8, Similar_matrix=consensus_similarity, Weight=weight)
    #         predict_labels = convert_to_label(predict_membership, noise_threshold=0.2)
    #     elif method_index == "OPTICS":
    #         from sklearn.cluster import OPTICS
    #         predict_labels = OPTICS(metric="precomputed", min_samples=10).fit_predict(consensus_similarity)
    #     elif method_index == "SC":
    #         from sklearn.cluster import SpectralClustering
    #         predict_labels = SpectralClustering(affinity="precomputed").fit_predict(consensus_similarity)
    #     elif method_index == "AP":
    #         from sklearn.cluster import AffinityPropagation
    #         predict_labels = AffinityPropagation(affinity="precomputed").fit_predict(consensus_similarity)
    #     # 计算精度，召回率
    #     from sklearn.metrics import normalized_mutual_info_score, accuracy_score, adjusted_rand_score, recall_score, \
    #         f1_score
    #     print("ACC:", accuracy_score(true_labels, predict_labels))
    #     print("NMI:", normalized_mutual_info_score(true_labels, predict_labels))
    #     print("F-Score:", f1_score(true_labels, predict_labels, average="macro"))
    #     print("ARI:", adjusted_rand_score(true_labels, predict_labels))
    #     print("Recall:", recall_score(true_labels, predict_labels, average="macro"))
    #     print("-" * 40)

    # 与其他文章的轨迹聚类进行比较（人工数据判断正确率，时间的比较）评估指标:各和总共SC，时间
    # 原算法
    # mvstfc = MV_STFC(len(track_id), file_path=os.path.join(os.getcwd(), "MV_STFC"))
    # mvstfc.Cluster(8, df, read_file=True)
    # mvstfc.Save_results(filepath=os.path.join(os.getcwd(), "MV_STFC"))
    # mvstfc.Evaluation(df)
    # 1.MTCA
    from MTCA import MTCA
    mtca = MTCA(len(track_id), file_path=os.path.join(os.getcwd(), "MTCA"))
    mtca.MFHD(df, keep_file=True, read_file=False)
    mtca = MTCA(len(track_id), file_path=os.path.join(os.getcwd(), "MTCA"))
    mtca.MFHD(df, keep_file=False, read_file=True)
    mtca.Cluster()
    mtca.Evaluation(df)
    mtca.Save_results(os.path.join(os.getcwd(), "MTCA"))
    # 2.DBTCAN

    # 3.MIF-STKNNDC
    # from MIF_STKNNDC import MIF_STKNNDC
    # stknndc = MIF_STKNNDC(len(track_id), file_path=os.path.join(os.getcwd(), "STKNNDC"))
    # # stknndc.STHD(df, keep_file=True, read_file=False)
    # stknndc.STHD(df, keep_file=False, read_file=True)
    # stknndc.FE_CTD(n_clusters=8)
    # stknndc.Evaluation(df)
    # stknndc.Save_results(os.path.join(os.getcwd(), "STKNNDC"))

    # 4.ISCM
    # from ISCM import ISCM
    # iscm = ISCM(len(track_id), file_path=os.path.join(os.getcwd(), "ISCM"))
    # iscm.Cluster(n_clusters=8, track_df=df, keep_file=True, read_file=False)
    # iscm = ISCM(len(track_id), file_path=os.path.join(os.getcwd(), "ISCM"))
    # iscm.Cluster(n_clusters=8, track_df=df, keep_file=False, read_file=True)
    # iscm.Evaluation(df)

    # 各种生成代表轨迹的方法（密度最高，DPC算法中最高的）
    # 遗传算法
    # 密度最大
    # DPC算法
