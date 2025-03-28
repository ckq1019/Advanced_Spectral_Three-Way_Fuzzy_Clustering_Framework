import pandas as pd
from Global_Variable import *
import numpy as np
import math


def dist(a, b):
    return math.sqrt(math.pow(b[1]-a[1], 2)+math.pow(b[0]-a[0], 2))


def p_line(s, e, p):
    '''
    求点p到由se组成的直线的垂直距离
    :param s: s[x1,y1]
    :param e: e[x2,y2]
    :param p: p[x3,y3]
    :return:
    '''
    if math.sqrt(pow((s[1]-e[1]), 2) + pow(s[0]-e[0], 2)) == 0:
        return 0
    else:
        return np.float64(math.fabs((s[1]-e[1])*p[0]+(e[0]-s[0])*p[1]+s[0]*e[1]-s[1]*e[0])/math.sqrt(
            (s[1]-e[1])**2 + (s[0]-e[0]) ** 2))


def Change_Trend_Similarity(tracka_point, trackb_point):
    '''
    计算两轨迹之间的变化趋势相似度
    :param TR_a:
    :param TR_b:
    :return:
    '''
    # 对各特征进行分别进行变化趋势
    Distance = []  # 把各特征的计算距离放入W=1
    for Character in range(3, tracka_point.shape[1]):
        # 进行轨迹分割
        from Trajectory_Segmentation.Trajectory_Segmentation_Methodology import myMethodology
        key_a = myMethodology(tracka_point[:, Character])  # 轨迹a的分割后的线段[slope, w(weight)]
        key_b = myMethodology(trackb_point[:, Character])

        DTW = {}  # 初始化DTW矩阵
        w = abs(len(key_a)-len(key_b))+1
        for i in range(len(key_a) - 1):
            DTW[(i, -1)] = float('inf')
        for i in range(len(key_b) - 1):
            DTW[(-1, i)] = float('inf')
        for i in range(len(key_a) - 1):
            DTW[(i, i + w)] = float('inf')
            DTW[(i, i - w - 1)] = float('inf')
        DTW[(-1, -1)] = 0
        weight_DTW = DTW  # 加权DTW
        # DTW计算
        for i in range(len(key_a) - 1):
            for j in range(max(0, i - w), min(len(key_b) - 1, i + w)):
                # 两个线段之间的距离（斜率之差*0.4+初末点之差*0.5*0.6）
                dist = (abs(key_a[i, 1] - key_b[j, 1]) + abs(key_a[i + 1, 1] - key_b[j + 1, 1])) * 0.5 * 0.6 + abs(
                    (key_a[i + 1, 1] - key_a[i, 1]) / (key_a[i + 1, 0] - key_a[i, 0]) -
                    (key_b[j + 1, 1] - key_b[j, 1]) / (key_b[j + 1, 0] - key_b[j, 0])) * 0.4
                Mininum = min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])  # 最小值定位
                if Mininum == DTW[(i - 1, j)]:
                    MiniPos = (i - 1, j)
                elif Mininum == DTW[(i, j - 1)]:
                    MiniPos = (i, j - 1)
                else:
                    MiniPos = (i - 1, j - 1)
                DTW[(i, j)] = dist + Mininum
                if i == 0 and j == 0:
                    continue
                weight_DTW[(i, j)] = dist * ((key_a[i + 1, 0] - key_a[i, 0]) / (key_a[-1, 0] - key_a[0, 0]) +
                                             (key_b[j + 1, 0] - key_b[j, 0]) / (key_b[-1, 0] - key_b[0, 0])) + \
                                     weight_DTW[MiniPos]
        Distance.append(weight_DTW[(len(key_a) - 2, len(key_b) - 2)])
    print("变化趋势相似度：", np.mean(Distance))
    return np.mean(Distance)  # W=1


def Timestamp_Similarity(tracka_point, trackb_point):
    # 计算两个轨迹之间的初始和结尾时间戳的相似度
    time = pd.to_datetime(tracka_point[0, 1], format="%Y%m%d")
    tra_start = time.day_of_year  # 开始时间
    tra_end = tra_start + np.ceil(tracka_point.shape[0] / 4)-1
    time = pd.to_datetime(trackb_point[0, 1], format="%Y%m%d")
    trb_start = time.day_of_year
    trb_end = trb_start + np.ceil(trackb_point.shape[0] / 4)-1  # 结束时间

    start_time = min([tra_start, trb_start])  # 两个轨迹最小的轨迹点
    end_time = max([tra_end, trb_end])  # 两个轨迹比较大的时间戳
    time_similar1 = 1 + (max(tra_start, trb_start) - min(tra_end, trb_end)) / (end_time - start_time)

    # 考虑会一个轨迹在年末一个轨迹在年初
    trb_start = trb_start-365
    trb_end = trb_end-365
    start_time = min([tra_start, trb_start])  # 两个轨迹最小的轨迹点
    end_time = max([tra_end, trb_end])  # 两个轨迹比较大的时间戳
    time_similar2 = 1 + (max(tra_start, trb_start) - min(tra_end, trb_end)) / (end_time - start_time)

    trb_start = trb_start + 365*2
    trb_end = trb_end + 365*2
    start_time = min([tra_start, trb_start])  # 两个轨迹最小的轨迹点
    end_time = max([tra_end, trb_end])  # 两个轨迹比较大的时间戳
    time_similar3 = 1 + (max(tra_start, trb_start) - min(tra_end, trb_end)) / (end_time - start_time)

    time_similar = min(time_similar1, time_similar2, time_similar3)
    print("时间相似度", time_similar)
    return time_similar


def Position_Similarity(tracka_point, trackb_point):
    '''
    计算两个轨迹之间(轨迹a和轨迹b)的初始和结尾位置相似度
    :param TR_a:
    :param TR_b:
    :return: w*轨迹间的垂直距离+w*两轨迹间的平行距离+w*两轨迹间的角距离， w=1/3
    '''
    L_a = [tracka_point[0][3:5].tolist(), tracka_point[-1][3:5].tolist()]
    L_b = [trackb_point[0][3:5].tolist(), trackb_point[-1][3:5].tolist()]

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


if __name__ == '__main__':
    file_path = os.path.join(DATA_PATH, r"hurdat2-1851-2021-100522.csv")
    if not os.path.exists(file_path):
        print("file not exists!")
    df = pd.read_csv(file_path)
    # 提取密集时间段-全部
    # 提取热点区域-全部
    # 相似度计算
    track_id = []  # 存放轨迹id
    track_array = []
    for track, track_df in df.groupby("track"):
        track_id.append(track)
        for i in track_df.index:
            track_array.append(list(track_df.loc[i]))
    track_array = np.array(track_array, dtype=np.float64)
    # 数据进行归一化
    for i in range(3, track_array.shape[1]):
        track_array[:, i] = (track_array[:, i] - min(track_array[:, i])) / (max(track_array[:, i]) - min(
            track_array[:, i]))
    similarity_matrix = np.zeros((len(track_id), len(track_id), 3))
    for i in range(len(track_id)):
        for j in range(i, len(track_id)):
            if i == j:
                continue
            tracka_point = track_array[track_array[:, 0] == track_id[i]]  # 轨迹a的（归一后的）点
            trackb_point = track_array[track_array[:, 0] == track_id[j]]  # 轨迹b的（归一后的）点
            # 轨迹变化趋势相似度
            similarity_matrix[j][i][0] = similarity_matrix[i][j][0] = Change_Trend_Similarity(tracka_point,
                                                                                              trackb_point)
            # 轨迹初末相似度计算
            similarity_matrix[j][i][1] = similarity_matrix[i][j][1] = Position_Similarity(tracka_point, trackb_point)
            # 轨迹时间戳相似度计算
            similarity_matrix[j][i][2] = similarity_matrix[i][j][2] = Timestamp_Similarity(tracka_point, trackb_point)
    np.save(os.path.join(Hurdat_PATH, "similarity_matrix.npy"), similarity_matrix)
    print("file saved successful.")
