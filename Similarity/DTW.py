import numpy as np
import math
import time
import os
import pandas as pd


def ACDTW(TR_a, TR_b, feature=None, cal_type='P', AC=False, W=None):
    '''
    计算轨迹TRa，TRb的多因子变化相似度, Multifactor_distance计算线段之间的距离（相似度）
    :param TR_a:
    :param TR_b:
    :param feature
    :param type计算相似度的对象类型：P：point ; L:line
    :param AC: TRUE:ACDTW ; FALSE:DTW
    :return:
    '''
    if cal_type == 'L':
        n, m = len(TR_a.line_list), len(TR_b.line_list)
    elif cal_type == 'P':
        # n, m = len(TR_a.point_list), len(TR_b.point_list)
        n, m = len(feature[feature['TrackId'] == float(TR_a.TrackId)].values), len(feature[feature['TrackId'] ==
                                                                                           float(TR_b.TrackId)].values)
    if AC:
        Q = np.zeros((n, m))  # 匹配过程中使用Q和C中的每个点的次数。
        C = np.zeros((n, m))
    if W is not None:
        W = max(W, abs(n-m))+1

    def Distance(i, j):
        # 计算i和j的距离
        from Similarity.Similarity_Measurement_Methodology import Multifactor_distance
        if cal_type == 'L':
            if feature is not None:
                dist = Multifactor_distance(list(feature[(feature['TrackId'] == float(TR_a.TrackId)) &
                                                                 (feature['LineId'] == float(i))].values[0]),
                                                         list(feature[(feature['TrackId'] == float(TR_b.TrackId)) &
                                                                 (feature['LineId'] == float(j))].values[0]))
                if TR_a.line_list[i].weight != 0 and TR_b.line_list[j].weight != 0:
                    dist *= (TR_a.line_list[i].weight + TR_b.line_list[j].weight)
            else:
                dist = Multifactor_distance(TR_a.line_list[i], TR_b.line_list[j])
        elif cal_type == 'P':
            if feature is not None:
                dist = Multifactor_distance(list(feature[feature['TrackId'] == float(TR_a.TrackId)].values)[i],
                                            list(feature[feature['TrackId'] == float(TR_b.TrackId)].values)[j])
            else:
                dist = Multifactor_distance(TR_a.point_list[i], TR_b.point_list[j])
        return dist

    dtw_matrix = np.full((n, m), np.inf)
    dtw_matrix[0, 0] = Distance(0, 0)
    if AC:
        def c(N):
            # 惩罚函数，N为该点在匹配过程中使用每个点的次数
            length = m/n if m <= n else n/m
            if length > 0.5:
                return 2*max(m, n)/(m+n) * N
            if length >0 and length <=0.5:
                return 0.5*(m+n)/max(m, n)*N
        result_matrix = dtw_matrix.copy()
    for i in range(0, n):
        if W is not None:
            x = range(max(0, i - W), min(m, i + W)) if i != 0 else range(1, min(m, i + W))
        else:
            x = range(0, m) if i != 0 else range(1, m)
        for j in x:
            if AC:
                D_ij = Distance(i, j)
                D1 = D_ij + c(C[i-1, j])*D_ij + dtw_matrix[i-1, j]
                D2 = D_ij + dtw_matrix[i-1, j-1]
                D3 = D_ij + c(Q[i, j-1])*D_ij + dtw_matrix[i, j-1]
                dtw_matrix[i, j] = min(D1, D2, D3)
                if dtw_matrix[i, j] == D1:
                    Q[i, j] = 1
                    C[i, j] = C[i-1, j] + 1
                    result_matrix[i, j] = D_ij + result_matrix[i-1, j]
                elif dtw_matrix[i, j] == D2:
                    Q[i, j] = 1
                    C[i, j] = 1
                    result_matrix[i, j] = D_ij + result_matrix[i - 1, j - 1]
                elif dtw_matrix[i, j] == D3:
                    Q[i, j] = Q[i, j-1] + 1
                    C[i, j] = 1
                    result_matrix[i, j] = D_ij + result_matrix[i, j - 1]
            else:
                dtw_matrix[i, j] = Distance(i, j) + min(dtw_matrix[i - 1, j], dtw_matrix[i - 1, j - 1],
                                                        dtw_matrix[i, j - 1])
    if cal_type == 'L':
        if AC:
            if TR_a.line_list[i].weight != 0 and TR_b.line_list[j].weight != 0:
                print("加权ACDTW距离相似度：", result_matrix[n - 1, m - 1])
                return dtw_matrix[n - 1, m - 1]
            print("ACDTW距离相似度：", result_matrix[n - 1, m - 1] / max(m, n))
            return dtw_matrix[n - 1, m - 1] / max(m, n)
        else:
            if TR_a.line_list[i].weight != 0 and TR_b.line_list[j].weight != 0:
                print("加权DTW距离相似度：", dtw_matrix[n-1, m-1])
                return dtw_matrix[n-1, m-1]
            else:
                # 因为可能会造成越长的轨迹DTW越长
                print("DTW距离相似度：", dtw_matrix[n-1, m-1]/max(m, n))
                return dtw_matrix[n-1, m-1]/max(m, n)
    elif cal_type == 'P':
        if AC:
            print("ACDTW距离相似度：", result_matrix[n - 1, m - 1] / max(m, n))
            return dtw_matrix[n - 1, m - 1] / max(m, n)
        else:
            # 因为可能会造成越长的轨迹DTW越长
            print("DTW距离相似度：", dtw_matrix[n-1, m-1]/max(m, n))
            return dtw_matrix[n-1, m-1]/max(m, n)
    else:
        print("未知格式计算！")
        return None


def distance(p1, p2):
    if type(p1) == np.float64:  # 如果p1不是列表的，而是某个数值的话，就直接计算
        return math.sqrt((p1 - p2) ** 2)
    x = 0
    for i in range(len(p1)):
        x += (p1[i] - p2[i]) ** 2
    return math.sqrt(x)


def DTW(s1, s2, windowSize, bestdist=None):
    DTW = {}
    w = max(windowSize, abs(len(s1)-len(s2))+1)
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    for i in range(len(s1)):
        DTW[(i, i+w)] = float('inf')
        DTW[(i, i-w-1)] = float('inf')

    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        if bestdist is not None:
            d = float('inf')
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist = distance(s1[i], s2[j])
            DTW[(i, j)] = dist + min(DTW[(i-1, j)], DTW[(i, j-1)], DTW[(i-1, j-1)])
            if bestdist is not None:
                if d > DTW[(i, j)]:
                    d = DTW[(i, j)]
        if bestdist is not None:
            if d >= bestdist:
                return d
    return DTW[(len(s1)-1, len(s2)-1)]


def LWDTW(track_a, track_b, windowSize, Weight=0.5):
    '''
    使用DTW对轨迹进行计算相似度（并对轨迹进行压缩）
    :param track_a: 轨迹a
    :param track_b: 轨迹b
    :param windowSize: 计算窗口
    :return:
    '''
    tracka_point = []  # 轨迹a的（归一后的）点
    trackb_point = []  # 轨迹b的（归一后的）点
    for p in track_a.point_list:
        tracka_point.append(p.array)
    for p in track_b.point_list:
        trackb_point.append(p.array)
    tracka_point = np.array(tracka_point, dtype=np.float64)
    trackb_point = np.array(trackb_point, dtype=np.float64)
    # 对各特征进行分别进行变化趋势
    # [2amplitude, 3x_coordinate, 4y_coordinate, 5speed_average, 6effective_radius, 7velocity, 8angle]
    Distance = []  # 把各特征的计算距离放入W=1
    for Character in range(2, 9):
        # 进行轨迹分割
        from Trajectory_Segmentation.Trajectory_Segmentation_Methodology import myMethodology
        key_a = myMethodology(tracka_point[:, Character])  # 轨迹a的得到的关键点[[0, x[0]], [index, x[index]], ...]
        key_b = myMethodology(trackb_point[:, Character])

        DTW = {}  # 初始化DTW矩阵
        w = max(windowSize, abs(len(key_a)-len(key_b))+1)
        for i in range(len(key_a)-1):
            DTW[(i, -1)] = float('inf')
        for i in range(len(key_b)-1):
            DTW[(-1, i)] = float('inf')
        for i in range(len(key_a)-1):
            DTW[(i, i+w)] = float('inf')
            DTW[(i, i-w-1)] = float('inf')
        DTW[(-1, -1)] = 0
        weight_DTW = DTW.copy()  # 加权DTW
        # DTW计算
        for i in range(len(key_a)-1):
            for j in range(max(0, i-w), min(len(key_b)-1, i+w)):
                # 两个线段之间的距离（斜率之差*0.4+初末点之差*0.5*0.6）
                dist = (abs(key_a[i, 1] - key_b[j, 1]) + abs(key_a[i+1, 1] - key_b[j+1, 1])) * 0.5 * (1-Weight) + \
                       abs((key_a[i+1, 1] - key_a[i, 1]) / (key_a[i+1, 0] - key_a[i, 0]) -
                           (key_b[j+1, 1] - key_b[j, 1]) / (key_b[j+1, 0] - key_b[j, 0])) * Weight
                Mininum = min(DTW[(i-1, j)], DTW[(i, j-1)], DTW[(i-1, j-1)])  # 最小值定位
                if Mininum == DTW[(i-1, j)]:
                    MiniPos = (i-1, j)
                elif Mininum == DTW[(i, j - 1)]:
                    MiniPos = (i, j - 1)
                else:
                    MiniPos = (i - 1, j - 1)
                DTW[(i, j)] = dist + Mininum
                # if i == 0 and j == 0:
                #     continue
                weight_DTW[(i, j)] = dist * ((key_a[i + 1, 0] - key_a[i, 0]) / (key_a[-1, 0] - key_a[0, 0]) +
                                             (key_b[j + 1, 0] - key_b[j, 0]) / (key_b[-1, 0] - key_b[0, 0])) + \
                                     weight_DTW[MiniPos]
        Distance.append(weight_DTW[(len(key_a)-2, len(key_b)-2)])
    print("变化趋势相似度：", np.mean(Distance))
    return np.mean(Distance)  # W=1


def getLB_oneQ_qbox(X, others, qbounds):
    '''
    Get the lower bounds between one query series X and many candidate series in others
    :param X: one series
    :param others: all candidate series
    :param qbounds: the bounding boxes of the query windows
    :return: the lower bounds between X and each candidate series
    '''
    lbs = []
    dim = len(X[0])
    for idy, s2 in enumerate(others):
        LB_sum = 0
        for idy, y in enumerate(s2):
            l = qbounds[idy][0]
            u = qbounds[idy][1]
            temp = math.sqrt(
                sum([(y[idd] - u[idd]) ** 2 if (y[idd] > u[idd]) else (l[idd] - y[idd]) ** 2 if (y[idd] < l[idd]) else 0
                     for idd in range(dim)]))
            LB_sum += temp
        lbs.append(LB_sum)
    return lbs


def LB_MV(TR_a, TR_b, feature, cal_type, W=0):
    '''
        Compute the DTW distance between a query series and a set of reference series.
        :param TR_a:计算轨迹a
        :param TR_b：
        :param feature:经过归一化的轨迹数据
        :param cal_type:计算对象类型
        :param W: half window size
        :return: the DTW distance and the coretime
        '''
    ditance = 0
    if feature is not None:
        query_a = np.array(feature[feature['TrackId'] == float(TR_a.TrackId)].values)
        query_b = np.array(feature[feature['TrackId'] == float(TR_b.TrackId)].values)
        n = len(query_a)
        m = len(query_b)
        dim = query_a.shape[1]
    else:
        if cal_type == 'L':
            # query_a, query_b代码待补充
            n = len(TR_a.line_list)
            m = len(TR_b.line_list)
        else:
            n = len(TR_a.point_list)
            m = len(TR_b.point_list)
    W = max(W, abs(n - m)) + 1
    for p, i in enumerate(query_a):
        lower_bound = min(query_b[max(0, p-W): min(len(query_b), p+W)])
        upper_bound = max(query_b[max(0, p-W): min(len(query_b), p+W)])
        if i >= upper_bound:
            ditance += (i-upper_bound) ** 2
        elif i < lower_bound:
            ditance += (i-lower_bound) ** 2
    math.sqrt(ditance)
    print("LB_MV距离：", ditance)
    return ditance

    bounds = []
    for idx in range(ql):
        segment = query[(idx - W if idx - W >= 0 else 0):(idx + W + 1 if idx + W <= ql - 1 else ql)]
        l = [min(segment[:, idd]) for idd in range(2, dim)]
        u = [max(segment[:, idd]) for idd in range(2, dim)]
        bounds.append([l, u])

    return


def TI_DTW(TH, P, query, references, W):
    '''
    Compute the TI_DTW distance between a query series and a set of reference series.
    :param i: the query ID number
    :param DTWdist: precomputed DTW distances (for fast experiments)
    :param TH: the triggering threshold for the expensive filter to take off
    :param query: the query series
    :param references: a list of reference series
    :param W: half window size
    :return: the DTW distance and the coretime
    '''
    skips = 0
    p_cals = 0
    coretime = 0

    start = time.time()
    # get bounds of query
    ql = len(query)
    dim = len(query[0])
    bounds = []
    for idx in range(ql):
        segment = query[(idx - W if idx - W >= 0 else 0):(idx + W + 1 if idx + W <= ql-1 else ql)]
        l = [min(segment[:, idd]) for idd in range(dim)]
        u = [max(segment[:, idd]) for idd in range(dim)]
        bounds.append([l, u])
    LBs = getLB_oneQ_qbox(query, references, bounds)
    LBSortedIndex = np.argsort(LBs)
    predId = LBSortedIndex[0]

    def DTWwnd(s1, s2, windowSize):
        '''
        Compute the DTW distance between s1 and s2, and also the neighbor distances of s1
        :param s1: a series
        :param s2: a series
        :param windowSize: half window size
        :return: DTW distance, neighbor distances
        '''
        DTW = {}
        dxx = []
        w = max(windowSize, abs(len(s1) - len(s2)))
        for i in range(len(s1)):
            DTW[(i, -1)] = float('inf')
        for i in range(len(s2)):
            DTW[(-1, i)] = float('inf')
        for i in range(len(s1)):
            DTW[(i, i + w)] = float('inf')
            DTW[(i, i - w - 1)] = float('inf')

        DTW[(-1, -1)] = 0

        for i in range(len(s1) - 1):
            dxx.append(distance(s1[i + 1], s1[i]))
            for j in range(max(0, i - w), min(len(s2), i + w)):
                dist = distance(s1[i], s2[j])
                DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])
        # final iteration
        i = len(s1) - 1
        for j in range(max(0, i - w), min(len(s2), i + w)):
            dist = distance(s1[i], s2[j])
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

        return DTW[len(s1) - 1, len(s2) - 1], dxx

    dist, dxx = DTWwnd(query, references[predId], W)

    def tiBounds_top_calP_list_comp_eb(X, Y, W, P, dxx, dist):
        # Same as tiBounds except that the true distances are calculated in every P samples of X
        # And early abondoning is used.
        Xlen = list(X.shape)[0]
        Ylen = list(Y.shape)[0]

        upperBounds = np.zeros([Xlen, W * 2 + 1])
        lowerBounds = np.zeros([Xlen, W * 2 + 1])

        lbrst = 0
        for t in range(0, Xlen):
            startIdx = 0 if t > W else W - t
            if t % P == 0:
                lw = max(0, t - W)
                tp = min(t + W + 1, Ylen)
                dxyInit = np.array([distance(X[t, :], Y[i, :]) for i in range(lw, tp)])

                upperBounds[t, startIdx:startIdx + tp - lw] = dxyInit
                lowerBounds[t, startIdx:startIdx + tp - lw] = dxyInit
                lbrst += np.amin(dxyInit)
            else:
                startIdx = 0 if t > W else W - t
                lr = 0 if t < W else t - W
                ur = Ylen - 1 if Ylen - 1 < t + W else t + W
                thisdxx = dxx[t - 1]
                startIdx_lr = startIdx - lr + 1
                t_1 = t - 1
                idx = ur - lr - 1
                if t + W <= Ylen - 1:
                    upperBounds[t, startIdx:startIdx + ur - lr] = [upperBounds[t_1, startIdx_lr + i] + thisdxx for i in
                                                                   range(lr, ur)]
                    lowerBounds[t, startIdx:startIdx + ur - lr] = \
                        [lowerBounds[t_1, startIdx_lr + i] - thisdxx if lowerBounds[t_1, startIdx_lr + i] > thisdxx
                         else 0 if thisdxx < upperBounds[t_1, startIdx_lr + i] else thisdxx - upperBounds[
                            t_1, startIdx_lr + i]
                         for i in range(lr, ur)]
                    # the last y point
                    temp = distance(X[t, :], Y[ur, :])
                    upperBounds[t, startIdx + idx + 1] = temp
                    lowerBounds[t, startIdx + idx + 1] = temp
                    lbrst += np.amin(lowerBounds[t, startIdx:startIdx + idx + 2])
                else:
                    upperBounds[t, startIdx:startIdx + idx + 2] = [upperBounds[t_1, startIdx_lr + i] + thisdxx for i in
                                                                   range(lr, ur + 1)]
                    lowerBounds[t, startIdx:startIdx + idx + 2] = \
                        [lowerBounds[t_1, startIdx_lr + i] - thisdxx if lowerBounds[t_1, startIdx_lr + i] > thisdxx
                         else 0 if thisdxx < upperBounds[t_1, startIdx_lr + i] else thisdxx - upperBounds[
                            t_1, startIdx_lr + i]
                         for i in range(lr, ur + 1)]
                    lbrst += np.amin(lowerBounds[t, startIdx:startIdx + idx + 2])
            if lbrst >= dist:
                return lbrst  # early abandoning
        return lbrst

    for x in range(1, len(LBSortedIndex)):
        thisrefid = LBSortedIndex[x]
        if LBs[thisrefid] >= dist:
            skips = len(LBs) - x
            break
        elif LBs[thisrefid] >= dist - TH*dist:
            p_lb = tiBounds_top_calP_list_comp_eb(query, references[thisrefid], P, W, dxx, dist)
            p_cals += 1
            if p_lb < dist:
                dist2 = DTW(query, references[thisrefid], W, dist)
                if dist > dist2:
                    dist = dist2
                    predId = thisrefid
            else:
                skips = len(LBs) - x
                break
        else:
            dist2 = DTW(query, references[thisrefid], W, dist)
            if dist > dist2:
                dist = dist2
                predId = thisrefid

    end = time.time()
    coretime += (end - start)

    return dist, predId, skips, coretime, p_cals


if __name__ == '__main__':
    file_name = os.path.join(r'D:\Trajectory_analysis\Data', 'Track_Compression.csv')
    data = pd.read_csv(file_name)
    from Data_Preprocessing import format_conversion
    tra_dic = format_conversion(data, file_format='CSV', Data_processing=False)
    # 不进行压缩轨迹，DTW计算两个整个轨迹的距离；运行时间（压缩时间，DTW运行时间），距离
    s1 = []
    s2 = []
    for p in tra_dic[37849].point_list:
        s1.append(p.array)
    for p in tra_dic[39017].point_list:
        s2.append(p.array)
    s1 = np.array(s1, dtype=np.float64)
    s2 = np.array(s2, dtype=np.float64)
    start = time.time()
    nocomprssion_value = DTW(s1[:, 2:], s2[:, 2:], 0) / max(s1.shape[0], s2.shape[0])
    end = time.time()
    spend_time = end - start
    print(spend_time)
    print("no split dtw:", nocomprssion_value)
    # 'MDL', 'WT', 'DP', 'ADPS', 'TrST', 'My_Method'
    # 从两个指标loss_compress(越小越好)和Compression_Accuracy（越大越好）两个指标看那个算法比较好
    from Cluster.Parameter_Selection_methodology import Loss_Compress, Compression_Accuracy
    # （1 - 差值 / 原本正确的值） / 压缩时间（压缩的时间 + DTW比较的时间），较小的时间，有较高的准确率
    from Trajectory_Segmentation.WT import normalized_slope_chart
    from Trajectory_Segmentation.Trajectory_Segmentation_Methodology import My_Method
    # print("1. 每个维度上提取关键点")
    # start = time.time()
    # value = 0
    # for i in range(2, s1.shape[-1]):
    #     x = s1[:, i]
    #     y = s2[:, i]
    #     s1_key = My_Method(x)
    #     s2_key = My_Method(y)
    #     if len(s1) - 1 not in s1_key:
    #         s1_key.append(len(s1) - 1)
    #     if len(s2) - 1 not in s2_key:
    #         s2_key.append(len(s2) - 1)
    #     s1_key = list(set(s1_key))
    #     s2_key = list(set(s2_key))
    #     s1_key.sort()
    #     s2_key.sort()
    #     value += DTW(s1[s1_key, i], s2[s2_key, i], 0) / max(len(s1_key), len(s2_key))
    # end = time.time()
    # print(end - start)
    # print("准确率（越小越好）：", abs(nocomprssion_value - value) / nocomprssion_value)
    # print("Compression_Accuracy(越大越好):", (1 - abs(nocomprssion_value - value) / nocomprssion_value) / (spend_time -
    #                                                                                                    (end - start)))
    print("--"*10)
    print("2. 所有维度上提取关键点")
    print("WT:")
    start = time.time()
    s1_key = []  # 保存s1的关键点index
    s2_key = []  # 保存s2的关键点index
    for i in range(2, s1.shape[-1]):
        x = s1[:, i]
        y = s2[:, i]
        s1_key.extend(normalized_slope_chart(x))
        s2_key.extend(normalized_slope_chart(y))
    if len(s1)-1 not in s1_key:
        s1_key.append(len(s1)-1)
    if len(s2) - 1 not in s2_key:
        s2_key.append(len(s2) - 1)
    s1_key = list(set(s1_key))
    s2_key = list(set(s2_key))
    s1_key.sort()
    s2_key.sort()
    value = DTW(s1[s1_key, 2:], s2[s2_key, 2:], 0) / max(len(s1_key), len(s2_key))
    end = time.time()
    print(end - start)
    print("准确率（越小越好）：", abs(nocomprssion_value - value) / nocomprssion_value)
    try:
        print("Compression_Accuracy(越大越好):", (1-abs(nocomprssion_value - value) / nocomprssion_value) / (spend_time -
                                                                                                         (end - start)))
    except:
        print("Compression_Accuracy(越大越好):inf")
    print("My_Method:")
    start = time.time()
    s1_key = []  # 保存s1的关键点index
    s2_key = []  # 保存s2的关键点index
    for i in range(2, s1.shape[-1]):
        x = s1[:, i]
        y = s2[:, i]
        s1_key.extend(My_Method(x))
        s2_key.extend(My_Method(y))
    if len(s1) - 1 not in s1_key:
        s1_key.append(len(s1) - 1)
    if len(s2) - 1 not in s2_key:
        s2_key.append(len(s2) - 1)
    s1_key = list(set(s1_key))
    s2_key = list(set(s2_key))
    s1_key.sort()
    s2_key.sort()
    value = DTW(s1[s1_key, 2:], s2[s2_key, 2:], 0) / max(len(s1_key), len(s2_key))
    end = time.time()
    print(end - start)
    print("准确率（越小越好）：", abs(nocomprssion_value - value) / nocomprssion_value)
    print("Compression_Accuracy(越大越好):", (1 - abs(nocomprssion_value - value) / nocomprssion_value) / (spend_time -
                                                                                                       (end - start)))
