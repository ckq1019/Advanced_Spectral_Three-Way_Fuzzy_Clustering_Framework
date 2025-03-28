import matplotlib.pyplot as plt
import pandas as pd

from Global_Variable import *
import numpy as np
import math
import time
from Similarity.DTW import ACDTW, DTW, LWDTW
from Similarity.HD import MFHD


def dist(a, b):
    return math.sqrt(math.pow(b[1]-a[1], 2)+math.pow(b[0]-a[0], 2))


def time_change(timeArray):
    # 把每个全部时间戳转成每年时间戳
    timeArray = pd.to_datetime('1950-01-01 00:00:00') + pd.Timedelta(str(timeArray) + 'D')
    t = timeArray - pd.to_datetime(str(timeArray.year) + '-01-01 00:00:00')
    return t.days


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
        return np.float(math.fabs((s[1]-e[1])*p[0]+(e[0]-s[0])*p[1]+s[0]*e[1]-s[1]*e[0])/math.sqrt(
            (s[1]-e[1])**2 + (s[0]-e[0]) ** 2))


def TD(TRa, TRb, delta=7):
    '''
    需要把时间戳转换格式, 废弃
    :param TRa: 轨迹a的时间跨度[trs,tre]
    :param TRb: 轨迹b的时间跨度[trs,tre]
    :param delta: 时间窗口
    :return: 两个轨迹的时间相似度的度量
    '''

    trs1 = time_change(TRa.TimeSpan[0])
    tre1 = time_change(TRa.TimeSpan[1])
    trs2 = time_change(TRb.TimeSpan[0])
    tre2 = time_change(TRb.TimeSpan[1])

    def lifespan(trs, tre):
        return abs(tre - trs)

    def intersection(span1, span2):
        # 两个时间段的重叠部分
        start = min(span1[0], span1[1], span2[0], span2[1])
        end = max(span1[0], span1[1], span2[0], span2[1])
        dur = (lifespan(span1[0], span1[1])+lifespan(span2[0], span2[1])) - lifespan(start, end)
        return dur if dur > 0 else 0

    def MDO():
        return max(intersection([trs1, tre1], [trs2, tre2]),
                   intersection([trs1, tre1], [trs2, tre2+delta]),
                   intersection([trs1, tre1], [trs2-delta, tre2]))
    time_similar = math.sqrt(pow(1-MDO()/(lifespan(trs1, tre1)+lifespan(trs2, tre2)-MDO()), 2))
    print("时间相似度：", time_similar)
    return time_similar


def MBR(Pa_start, Pa_end, Pb_start, Pb_end, delta=0.4):
    # 计算初末点的MBR（高维>2）距离

    def intersection(a_start, a_end, b_start, b_end):
        # 计算重叠部分
        start = min(a_start, a_end, b_start, b_end)
        end = max(a_start, a_end, b_start, b_end)
        dur = (abs(a_end - a_start) + abs(b_end - b_start)) - abs(end - start)
        if dur > 0:
            return dur
        if b_start > b_end:
            b_start += delta
        else:
            b_end += delta
        start = min(a_start, a_end, b_start, b_end)
        end = max(a_start, a_end, b_start, b_end)
        dur = (abs(a_end - a_start) + abs(b_end - b_start)) - abs(end - start)
        if dur > 0:
            return dur

        if b_start > b_end:
            b_end -= delta
        else:
            b_start -= delta
        start = min(a_start, a_end, b_start, b_end)
        end = max(a_start, a_end, b_start, b_end)
        dur = (abs(a_end - a_start) + abs(b_end - b_start)) - abs(end - start)
        if dur > 0:
            return dur
        return 0

    for i in range(len(Pb_start)):
        if (Pa_end[i] - Pa_start[i]) * (Pb_end[i] - Pb_start[i]) < 0:
            # 变化方向不同
            return 0
        if intersection(Pa_start[i], Pa_end[i], Pb_start[i], Pb_end[i]) == 0:
            return 0
    return 1


def Multifactor_distance(a, b, cal_type='L', W=None):
    '''
    多因子距离计算
    :param a:
    :param b:
    :param cal_type: 计算对象类型
    :param W: 是否加权,目前就PCA数据需要加权,如果是的话W为权值
    :return:
    '''
    if W is not None:
        if cal_type == 'L':
            distance = 0
            for i in range(2, len(a)):
                distance = distance + W[i-2]*(a[i]-b[i])**2
            distance = math.sqrt(distance)
            return distance
        else:
            print('未知格式!!')
            return 0
    else:
        if cal_type == 'P' or cal_type == 'L':
            return np.linalg.norm(np.array(a[2:]) - np.array(b[2:]), 2)
        elif cal_type == 'Point':
            # 直接输入高维数据
            return np.linalg.norm(np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64), 2)
        else:
            print('未知格式！')
            return 0


def Change_Trend_Similarity(TR_a, TR_b, cal_type, algorithm, compression=True):
    '''
    计算两轨迹之间的变化趋势相似度
    :param TR_a:
    :param TR_b:
    :param cal_type: 计算对象
    :param algorithm: 计算算法：DTW, ACDTW, HF, LIP_DTW
    :param compression: 进行轨迹压缩的算法
    :return:
    '''
    # 方法：各特征分开（压缩）讨论
    if algorithm == 'DTW' and cal_type == "L":
        return LWDTW(TR_a, TR_b, windowSize=0)
    from Trajectory_Segmentation.Trajectory_Segmentation_Methodology import trajectory_segment
    feature_matrix = []  # 存储关键点 or 分割段
    feature_matrix.extend(trajectory_segment(TR_a, draw=False, algorithm=compression, cal_type=cal_type,
                                             PCA_precess=False))
    feature_matrix.extend(trajectory_segment(TR_b, draw=False, algorithm=compression, cal_type=cal_type,
                                             PCA_precess=False))
    if cal_type == 'L':
        df = pd.DataFrame(data=feature_matrix, columns=['TrackId', 'LineId', 'amplitude_v', 'x_v', 'y_v',
                                                        'speed_v', 'radius_v', 'velocity_v', 'angle_v'])
    elif cal_type == 'P':
        df = pd.DataFrame(data=feature_matrix, columns=['TrackId', 'time', 'amplitude', 'x', 'y',
                                                        'speed', 'radius', 'velocity', 'angle'])
    # 计算两轨迹之间的变化趋势相似度
    if algorithm == 'DTW':
        return ACDTW(TR_a, TR_b, df, cal_type, AC=False, W=0)
    elif algorithm == 'ACDTW':
        return ACDTW(TR_a, TR_b, df, cal_type, AC=True, W=0)
    elif algorithm == 'HD':
        return MFHD(TR_a, TR_b, df, cal_type)


def Timestamp_Similarity(TR_a, TR_b):
    # 计算两个轨迹之间的初始和结尾时间戳的相似度
    # 这种计算方式不可靠！比如d[(1,4)->(2,3)] == d[(1,4)->(2,5)]
    # time_similar = abs(
    #             time_change(TR_a.TimeSpan[0]) - time_change(TR_b.TimeSpan[0])) * 0.5 + abs(
    #             time_change(time_change(TR_a.TimeSpan[1]) - time_change(TR_b.TimeSpan[1]))) * 0.5
    # time_similar = time_similar * 0.5 + abs((TR_a.TimeSpan[1] - TR_a.TimeSpan[0]) -
    #                                         (TR_b.TimeSpan[1] - TR_b.TimeSpan[0])) * 0.5
    # print("时间相似度", time_similar)
    if time_change(TR_a.TimeSpan[0]) > time_change(TR_a.TimeSpan[1]):  # TR_a可能是跨年轨迹
        tra_start = time_change(TR_a.TimeSpan[0])
        tra_end = tra_start + (TR_a.TimeSpan[1] - TR_a.TimeSpan[0])
    else:
        tra_start = time_change(TR_a.TimeSpan[0])
        tra_end = time_change(TR_a.TimeSpan[1])

    if time_change(TR_b.TimeSpan[0]) > time_change(TR_b.TimeSpan[1]):  # TR_b可能是跨年轨迹
        trb_start = time_change(TR_b.TimeSpan[0])
        trb_end = trb_start + (TR_b.TimeSpan[1] - TR_b.TimeSpan[0])
    else:
        trb_start = time_change(TR_b.TimeSpan[0])
        trb_end = time_change(TR_b.TimeSpan[1])

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


def Position_Similarity(TR_a, TR_b):
    '''
    计算两个轨迹之间(轨迹a和轨迹b)的初始和结尾位置相似度
    :param TR_a:
    :param TR_b:
    :return: w*轨迹间的垂直距离+w*两轨迹间的平行距离+w*两轨迹间的角距离， w=1/3
    '''
    # pos_similar = np.linalg.norm(np.array(TR_a.point_list[0].array[3:5], dtype=np.float64) -
    #                       np.array(TR_b.point_list[0].array[3:5], dtype=np.float64), 2) * 0.5 +\
    #        np.linalg.norm(np.array(TR_a.point_list[-1].array[3:5], dtype=np.float64) -
    #                       np.array(TR_b.point_list[-1].array[3:5], dtype=np.float64), 2) * 0.5
    # print("轨迹初末点空间相似度：", pos_similar)
    # return pos_similar
    # return Multifactor_distance(TR_a.point_list[0].array, TR_b.point_list[0].array, cal_type='P') * 0.5 + \
    #                                      Multifactor_distance(TR_a.point_list[-1].array,
    #                                                           TR_b.point_list[-1].array, cal_type='P') * 0.5

    L_a = [TR_a.point_list[0].array[3:5], TR_a.point_list[-1].array[3:5]]
    L_b = [TR_b.point_list[0].array[3:5], TR_b.point_list[-1].array[3:5]]

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
        # La_angle = np.arctan2(L_a[1][1] - L_a[0][1], L_a[1][0] - L_a[0][0]) * 180 / np.pi
        # Lb_angle = np.arctan2(L_b[1][1] - L_b[0][1], L_b[1][0] - L_b[0][0]) * 180 / np.pi
        # angle = 360 - abs(La_angle - Lb_angle) if abs(La_angle - Lb_angle) > 180 else abs(La_angle - Lb_angle)
        # if angle > 90:
        #     return math.fabs(a + b)
        # else:
        #     return math.fabs(a - b)

    pos_similar = math.sqrt(d_prp() + d_prl() + d_angle())
    # pos_similar = ((L_a[0][0] - L_b[0][0]) ** 2 + (L_a[1][0] - L_b[1][0]) ** 2) / (
    #             (L_a[0][0] - L_b[0][0]) + (L_a[1][0] - L_b[1][0])) / 3 + (
    #             (L_a[0][1] - L_b[0][1]) ** 2 + (L_a[1][1] - L_b[1][1]) ** 2) / (
    #             (L_a[0][1] - L_b[0][1]) + (L_a[1][1] - L_b[1][1])) / 3 + (
    #             (L_a[0][0] - L_b[0][0]) ** 2 + (L_a[1][0] - L_b[1][0]) ** 2) / 3
    print("轨迹初末点空间相似度：", pos_similar)
    return pos_similar


def Similarity_Matrix(TR_List, cal_type='P', isCalculate=False, algorithm='DTW', Progress_bar=False, Compression=True):
    '''
    compute Similarity matrix
    :param TR_List: 轨迹列表
    :param cal_type: 计算相似度对象：点与点之间的或是段与段
    :param isCalculate:是否需要计算相似度，如果否的话就直接从文件中读取数据
    :param algorithm:算法选择：DTW;HD:Hausdorff distance
    :param Progress_bar:进度条是否可视化
    :param Compression:是否进行轨迹压缩
    :return:
    '''
    if not isCalculate:
        if not os.path.exists(os.path.join(DATA_PATH, 'Similarity_matrix.npy')):
            print("{} file is not exits!please check!".format(os.path.join(DATA_PATH, 'Similarity_matrix.npy')))
        file = os.path.join(DATA_PATH, 'Similarity_matrix.npy')
        similarity_matrix = np.load(file)  # 上邻接矩阵，[轨迹变化相似度， 轨迹出发结束位置点，时间戳距离]
        n = similarity_matrix.shape[-1]
        if len(TR_List) > 2:
            # 根据KNN算法去噪声, K取2，//Grubbs检验（最大标准残差检验）
            for i in range(n):
                similarity_matrix[:, :, i] += similarity_matrix[:, :, i].T
                # similarity_matrix[:, :, i][similarity_matrix[:, :, i] == 0] = np.nan  # 以防归一化后会有影响
                # series = []  # 存放第K的距离，去除噪声
                # for j in range(len(TR_List)):
                #     k_index = np.argsort(similarity_matrix[:, :, i][j])[2]
                #     series.append(similarity_matrix[:, :, i][j][k_index])
                # up_limit = np.mean(series) + 3 * np.std(k_index)
                # down_limit = np.mean(series) - 3 * np.std(k_index)
                # similarity_matrix[:, :, i][similarity_matrix[:, :, i] > up_limit] = np.nan
                # similarity_matrix[:, :, i][similarity_matrix[:, :, i] < down_limit] = np.nan
            # 相似度归一化
            # for i in range(n):
                # s = [min(similarity_matrix[:, :, i][similarity_matrix[:, :, i].nonzero()].flat)] * len(TR_List) * \
                #     len(TR_List)
                # s = [min(similarity_matrix[:, :, i].flat)] * len(TR_List) * len(TR_List)
                # b = [max(similarity_matrix[:, :, i].flat)] * len(TR_List) * len(TR_List)
                # s = np.array(s, dtype=np.float64)
                # b = np.array(b, dtype=np.float64)
                # s = s.reshape((len(TR_List), len(TR_List)))
                # b = b.reshape((len(TR_List), len(TR_List)))
                # similarity_matrix[:, :, i] = (similarity_matrix[:, :, i] - s) / (b - s)
                # similarity_matrix[:, :, i][similarity_matrix[:, :, i] < 0] = 0
        # similarity_matrix[:, :, 0] = similarity_matrix[:, :, 0] * 0.5 + similarity_matrix[:, :, 2] * 0.5
        # 对数据进行中的噪声轨迹进行处理,并对相似度矩阵进行处理
        delete = []
        delete_index = []
        for i in range(len(TR_List)):
            if all(np.isnan(similarity_matrix[i, :, 0])):
                    TR_List[i].noise = True
                    delete.append(TR_List[i])
                    delete_index.append(i)
        # 删除噪声轨迹TR_list,similarity_matrix
        print("删除{}条噪声轨迹：".format(len(delete_index)), delete_index)
        for tr in delete:
            TR_List.remove(tr)
        similarity_matrix = np.delete(similarity_matrix, delete_index, 0)
        similarity_matrix = np.delete(similarity_matrix, delete_index, 1)
        # delete = []
        # delete_index = []
        # for tr in delete:
        #     TR_List.remove(tr)
        # similarity_matrix = np.delete(similarity_matrix, delete_index, 0)
        # similarity_matrix = np.delete(similarity_matrix, delete_index, 1)
        return similarity_matrix

    similarity_matrix = np.zeros([len(TR_List), len(TR_List), 3], dtype=np.float64)  # [轨迹变化相似度， 轨迹出发结束位置点，时间戳距离]
    if Progress_bar:
        all_task = len(TR_List) * len(TR_List) / 2
        current_task = 0
        print('-' * current_task, '> {}%'.format(round(current_task/all_task, 4) * 100))
    for i in range(len(TR_List)):
        for j in range(i + 1, len(TR_List)):
            # 轨迹初末相似度计算
            # if not MBR(TR_List[i].point_list[0].array[2:], TR_List[i].point_list[-1].array[2:],
            #                             TR_List[j].point_list[0].array[2:], TR_List[j].point_list[-1].array[2:]):
            #     continue
            similarity_matrix[i][j][1] = Position_Similarity(TR_List[i], TR_List[j])
            # 轨迹时间戳相似度计算
            # time_similar = TD(TR_List[i], TR_List[j])
            # if time_similar == 1:  # 如果是1就说明没有重叠部分
            #     continue
            similarity_matrix[i][j][2] = Timestamp_Similarity(TR_List[i], TR_List[j])
            # 轨迹变化趋势相似度
            similarity_matrix[i][j][0] = Change_Trend_Similarity(TR_List[i], TR_List[j], cal_type, algorithm,
                                                                 Compression)
            if Progress_bar:
                current_task += 1
                print('-' * min(20, int(current_task / all_task)), '> {}%'.format(round(current_task / all_task,
                                                                                        4) * 100))
        # similarity_matrix[:, :, 0] += similarity_matrix[:, :, 0].T
        # if len(similarity_matrix[i, similarity_matrix[i, :, 0] != 0, 0]) == 0:  # 如果没有相邻点，记为噪声
        #     continue
        # while len(similarity_matrix[i, similarity_matrix[i, :, 0] != 0, 0]) < K:  # 轨迹i到其他轨迹存在K个
        #     print(len(similarity_matrix[i, similarity_matrix[i, :, 0] != 0, 0]))
        #     zero_index = np.where(similarity_matrix[i, :, 0] == 0)[0]
        #     for j in zero_index:
        #         if j != i:  # 找到非i值的零值的index
        #             break
        #     similarity_matrix[i][j][0] = Change_Trend_Similarity(TR_List[i], TR_List[j], feature, cal_type, algorithm)
        #     similarity_matrix[i][j][1] = Position_Similarity(TR_List[i], TR_List[j])
        #     similarity_matrix[i][j][2] = TD(TR_List[i], TR_List[j], 15)
    # 存储相似性矩阵
    np.save(os.path.join(OUTPUT_PATH, 'Similarity_matrix.npy'), similarity_matrix)
    # np.savetxt(os.path.join(OUTPUT_PATH, 'Similarity_matrix.txt'), result_matrix, fmt='%f', delimiter=',')
    print("Similarity_matrix.npy 存储完毕")
    return similarity_matrix


class SimilarityMatrix:
    # 计算轨迹相似度
    def __init__(self, isCalculate, TR_List, K=5, filepath=os.path.join(SIMILAR_MATRIX, 'Similarity_matrix.npy'),
                 reshuffle=False):
        self.similarity_matrix = None
        if not isCalculate:
            self.K = K
            if not os.path.exists(filepath):  # 存储相似度矩阵的地址
                print("{} file is not exits!please check!".format(filepath))
            if reshuffle:
                print("Start processing Similarity_matrix :{}".format(filepath))
                self.handle_matrix(filepath)
                self.savefile(os.path.join(os.path.dirname(filepath), 'KNN_Similarity_matrix.npy'))
            else:
                self.similarity_matrix = np.load(filepath)  # 直接读取文件中的数据
                print(" {} file read.".format(filepath))
                print(max(self.similarity_matrix[:, :, 0].flat))
                print(max(self.similarity_matrix[:, :, 1].flat))
                print(max(self.similarity_matrix[:, :, 2].flat))
                # 进行数据扩大到一个范围
                print("Similarity_matrix normalization.")
                # if len(self.similarity_matrix[self.similarity_matrix > 1]) > 0:
                #     # 进行数据归一化
                #     for i in range(self.similarity_matrix.shape[-1]):
                #         self.similarity_matrix[:, :, i] = (self.similarity_matrix[:, :, i] - min(
                #             self.similarity_matrix[:, :, i].flat)) / (max(self.similarity_matrix[:, :, i].flat) - min(
                #             self.similarity_matrix[:, :, i].flat))
                #     print("Similarity_matrix normalization.")
        else:
            # [轨迹变化相似度， 轨迹出发结束位置点，时间戳距离]
            print("Start calculate similar matrix.")
            self.similarity_matrix = np.zeros([len(TR_List), len(TR_List), 3], dtype=np.float64)
            self.TrackList = TR_List  # 轨迹列表
            self.K = K if K != 0 else len(TR_List)-1  # 计算邻接K个相似度
            self.all_task = len(TR_List) * self.K  # 所有的计算任务
            self.current_task = 0  # 当前任务数
            self.ProgressBar()

    def handle_matrix(self, filepath, MAX=np.inf):
        similarity_matrix = np.load(filepath)
        # 对相似度矩阵进行处理:把除了KNN距离，和自身距离，其他都定义为MAX
        n = similarity_matrix.shape[0]
        for i in range(n):  # 对每条轨迹找到最小距离的轨迹初末点时空距离
            for j in range(n):
                if j != i and similarity_matrix[i, j, 0] == 0:
                    similarity_matrix[i, j, :] = np.array([MAX, MAX, MAX])
            # k = []  # 存放最近的k个时空距离的index
            # distance = similarity_matrix[i, :, 1] ** 2 + similarity_matrix[i, :, 2] ** 2  # 记录初末点时空距离
            # for j in np.argsort(distance):
            #     if distance[j] == 0:
            #         continue
            #     if len(k) >= self.K:
            #         break
            #     k.append(j)
            # for j in range(n):
            #     if j not in k and j != i:
            #         similarity_matrix[i, j, :] = np.array([MAX, MAX, MAX])
            # if len(k) < self.K:
            #     print("{} track is less than K, please check!".format(str(i)))
        # 用dijkstra补全相似度矩阵
        for i in range(similarity_matrix.shape[-1]):
            for j in range(n):
                similarity_matrix[j, :, i] = self.dijkstra(similarity_matrix[:, :, i], j)
        # 对每个特征进行归一化
        for i in range(similarity_matrix.shape[-1]):
            similarity_matrix[:, :, i] = (similarity_matrix[:, :, i] - min(similarity_matrix[:, :, i].flat)) / (max(
                similarity_matrix[:, :, i].flat) - min(similarity_matrix[:, :, i].flat))
        self.similarity_matrix = similarity_matrix

    def calculation(self, tra_index, neighbour_index, cal_type, algorithm, Compression='WT'):
        '''
        计算轨迹Track_a与邻接轨迹的相似度（距离）
        :param tra_index:
        :param neighbour_index: [index1, index2,...]
        :param cal_type: 计算相似度对象
        :param algorithm: 计算轨迹变化的算法
        :param Compression: 进行轨迹压缩的算法[WT, ADPS,DP, TrST, None]
        :return:
        '''
        for neighbour in neighbour_index:  # 遍历所有邻接
            self.current_task += 1
            if neighbour == tra_index or self.similarity_matrix[tra_index][neighbour][0] != 0:
                continue
            # 轨迹变化趋势相似度
            self.similarity_matrix[neighbour][tra_index][0] = self.similarity_matrix[tra_index][neighbour][0] = \
                Change_Trend_Similarity(self.TrackList[tra_index], self.TrackList[neighbour], cal_type, algorithm,
                                        Compression)
            # 轨迹初末相似度计算
            self.similarity_matrix[neighbour][tra_index][1] = self.similarity_matrix[tra_index][neighbour][1] = \
                Position_Similarity(self.TrackList[tra_index], self.TrackList[neighbour])
            # 轨迹时间戳相似度计算
            self.similarity_matrix[neighbour][tra_index][2] = self.similarity_matrix[tra_index][neighbour][2] = \
                Timestamp_Similarity(self.TrackList[tra_index], self.TrackList[neighbour])
            self.ProgressBar()
        self.current_task = int((tra_index+1) / self.similarity_matrix.shape[0] * self.all_task )

    def savefile(self, path=None):
        if path is None:
            path = os.path.join(SIMILAR_MATRIX, 'Similarity_matrix.npy')
        np.save(path, self.similarity_matrix)
        print("{} file saved!".format(path))

    def ProgressBar(self):
        print('-' * min(30, int(self.current_task / self.all_task * 100)), '> {}%'.format(round(
            self.current_task / self.all_task, 4) * 100))

    def dijkstra(self, matrix, start_node):
        matrix_length = len(matrix)  # 矩阵一维数组的长度，即节点的个数
        used_node = [False] * matrix_length  # 访问过的节点数组
        distance = [np.inf] * matrix_length  # 最短路径距离数组
        distance[start_node] = 0  # 初始化，将起始节点的最短路径修改成0
        # 将访问节点中未访问的个数作为循环值，其实也可以用个点长度代替。
        while used_node.count(False):
            min_value = np.inf
            min_value_index = -1

            # 在最短路径节点中找到最小值，已经访问过的不在参与循环。
            # 得到最小值下标，每循环一次肯定有一个最小值
            for index in range(matrix_length):
                if not used_node[index] and distance[index] < min_value:
                    min_value = distance[index]
                    min_value_index = index

            # 将访问节点数组对应的值修改成True，标志其已经访问过了
            used_node[min_value_index] = True
            # 更新distance数组。
            # 以B点为例：distance[x] 起始点达到B点的距离。
            # distance[min_value_index] + matrix[min_value_index][index] 是起始点经过某点达到B点的距离，比较两个值，取较小的那个。
            for index in range(matrix_length):
                distance[index] = min(distance[index], distance[min_value_index] + matrix[min_value_index][index],
                                      distance[min_value_index] + matrix[index][min_value_index])
        return distance


if __name__ == '__main__':
    file_path = 'cyclonic_[353  80].csv'
    from main import read_data
    origin_data, format_file = read_data(os.path.join(r'D:\Trajectory_analysis\Data', file_path))
    from Data_Preprocessing import format_conversion
    data = format_conversion(origin_data, file_format='CSV', year_range=[2018, 2022], latitude_range=[0, 25.0],
                             longitude_range=[100.0, 122.0], Data_processing=False)
    tr_list = []
    for i in data.keys():
        if len(tr_list) == 0:
            tr_list.append(data[i])
            continue
        if data[i].NumPoint != tr_list[0].NumPoint:
            continue
        tr_list.append(data[i])

    sim_mat = np.zeros((3, 3))  # 四个轨迹
    track0_point = []
    track0_line = []
    for j, p in enumerate(tr_list[0].point_list):
        track0_point.append(p.array)
        if j == 0:
            continue
        track0_line.append(np.array(p.array, dtype=np.float64) - np.array(
            tr_list[0].point_list[j-1].array, dtype=np.float64))
    track0_point = np.array(track0_point, dtype=np.float64)
    track0_line = np.array(track0_line, dtype=np.float64)
    for i in range(1, 4):
        track_point = []
        track_line = []
        for j, p in enumerate(tr_list[i].point_list):
            track_point.append(p.array)
            if j == 0:
                continue
            track_line.append(np.array(p.array, dtype=np.float64) - np.array(tr_list[i].point_list[j-1].array,
                                                                             dtype=np.float64))
        track_point = np.array(track_point, dtype=np.float64)
        track_line = np.array(track_line, dtype=np.float64)

        x = []
        for char in range(2, 9):
            x.append(DTW(track0_line[:, char], track_line[:, char], 0, bestdist=None))

        sim_mat[i - 1, 0] = np.mean(x)  # DTW段
        sim_mat[i - 1, 1] = DTW(track0_point[:, 2:], track_point[:, 2:], 0, bestdist=None)  # DTW点
        # sim_mat[i - 1, 2] = MFHD()  # DTW点

    corr = []
    correlation0 = np.corrcoef(track0_point[:, 1:].T)[0]
    for i in range(1, 4):
        point = []
        for j, p in enumerate(tr_list[i].point_list):
            point.append(p.array)
        point = np.array(point, dtype=np.float64)

        track_corr = np.corrcoef(point[:, 1:].T)[0]
        corr.append(round(np.mean((correlation0 - track_corr) ** 2), 8))
        print("track0-{}(越小越好):".format(i), corr[-1])

    plt.figure()
    plt.plot(np.arange(sim_mat.shape[0]), sim_mat[:, 0], "r")
    plt.title("DTW line")

    plt.figure()
    plt.plot(np.arange(sim_mat.shape[1]), sim_mat[:, 1], "b")
    plt.title("DTW point")

    plt.figure()
    plt.plot(np.arange(sim_mat.shape[0]), np.array(corr, dtype=np.float64), "g")
    plt.title("corrcoef")
    plt.show()

    fig = plt.figure(figsize=(25, 2))
    ax = fig.subplots(3, 7)  # [track 0, track 3, track 9]
    for i in range(3):
        point = []
        for j, p in enumerate(tr_list[i+1].point_list):
            point.append(p.array)
        point = np.array(point, dtype=np.float64)
        for j in range(2, 9):
            ax[i, j-2].plot(np.arange(len(point[:, j])), point[:, j], color='green')
            ax[i, j - 2].plot(np.arange(len(track0_point[:, j])), track0_point[:, j], color='red')
    plt.show()

    # cal_type = 'P'  # 'P' or 'L'
    # tr_list = []
    # feature_matrix = []
    # from Trajectory_Segmentation.Trajectory_Segmentation_Methodology import trajectory_segment
    # start = time.time()
    # for i in data.keys():
    #     feature_matrix.extend(
    #         trajectory_segment(data[i], draw=False, algorithm='ADPS', cal_type=cal_type, PCA_precess=False))
    #     tr_list.append(data[i])
    # feature_matrix = np.array(feature_matrix, dtype=np.float64)
    # # if cal_type == 'L':
    # #     df = pd.DataFrame(data=feature_matrix, columns=['TrackId', 'LineId', 'amplitude_v', 'x_v', 'y_v',
    # #                                                     'speed_v', 'radius_v', 'velocity_v', 'angle_v'])
    # # elif cal_type == 'P':
    # df = pd.DataFrame(data=feature_matrix, columns=['TrackId', 'time', 'amplitude', 'x', 'y',
    #                                                 'speed', 'radius', 'velocity', 'angle'])
    # ACDTW(tr_list[0], tr_list[1], df, cal_type, AC=False)
    # end = time.time()
    # print(end - start)
    # start = time.time()
    # ACDTW(tr_list[0], tr_list[1], df, cal_type, AC=True)
    # end = time.time()
    # print(end - start)
    # start = time.time()
    # ACDTW(tr_list[0], tr_list[1], df, cal_type, AC=False, W=0)
    # end = time.time()
    # print(end - start)
    # start = time.time()
    # ACDTW(tr_list[0], tr_list[1], df, cal_type, AC=True, W=0)
    # end = time.time()
    # print(end - start)
    #
