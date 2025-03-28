import os
from Global_Variable import OUTPUT_PATH
import numpy as np
import matplotlib.pyplot as plt
from pyecharts.charts import Line
import pyecharts.options as opts
from Data_Analysis.Data_Visualization import open_web
from sklearn.datasets import make_blobs
import time


def optics_parameters(dist_matrix, k, draw=False):
    '''
    根据可达距离（kth作为参考点）
    :param dist_matrix:距离矩阵
    :param k:第k值作为测量可达距离
    :param draw:是否显示图
    :return:
    '''
    dist_matrix_c = dist_matrix.copy()
    k_distance = []
    p = 0
    for i in range(0, dist_matrix_c.shape[0]):
        index = []
        for j in np.argsort(dist_matrix_c[p]):
            if j in np.nonzero(dist_matrix_c[p])[0]:
                index.append(j)
        k_index = index[k] if k < len(index) else None
        if k_index == None:
            break
        k_distance.append(dist_matrix_c[p][k_index])
        dist_matrix_c[p].fill(0)
        dist_matrix_c[:, p].fill(0)
        p = k_index
    # plt.figure()
    # plt.plot(np.arange(len(k_distance)), k_distance)
    # plt.draw()
    # plt.savefig(os.path.join(OUTPUT_PATH, "optics_K_"+str(k)+".png"))
    # print("optics_K_"+str(k)+".png 保存成功！")
    # plt.show()
    # plt.close()
    line = Line(opts.InitOpts(width="900px", height="500px"))
    line.add_xaxis([i for i in range(len(k_distance))])
    line.add_yaxis('similar_distance', y_axis=k_distance)
    line.set_global_opts(legend_opts=opts.LegendOpts(is_show=False),
                         xaxis_opts=opts.AxisOpts(type_="value"),
                         yaxis_opts=opts.AxisOpts(type_="value", min_=min(k_distance), max_=max(k_distance),
                                                  splitline_opts=opts.SplitLineOpts(is_show=True)),
                         toolbox_opts=opts.ToolboxOpts())
    path = os.path.join(OUTPUT_PATH, 'optics_K_'+str(k)+'_optics.html')
    line.render(path)
    print(path+" 文件存储完毕！")
    open_web(path)


def Evaluation_Model(result_cluster, similar_matrix, point_matrix, label_, except_noise=False):
    # 评估模型
    if similar_matrix is not None:
        if -1 in result_cluster.keys():
            result_cluster.pop(-1)
        if len(similar_matrix.shape) == 3:
            similar_matrix = similar_matrix[:, :, 0] ** 2 + similar_matrix[:, :, 1] ** 2 + similar_matrix[:, :, 2] ** 2
        return [SC(result_cluster, similar_matrix), CHI(result_cluster, similar_matrix),
                DVI(result_cluster, similar_matrix)]

    if label_ is None and result_cluster is not None:
        label_ = np.zeros((similar_matrix.shape[0]))
        for c in result_cluster.keys():
            if c == 'noise':
                continue
            for i in result_cluster[c]:
                label_[i] = c
        label_ = label_.tolist()
    print("K(包括噪声):", len(set(label_)))
    if except_noise:
        if 'noise' in result_cluster.keys():
            part_point_matrix = np.delete(point_matrix, result_cluster['noise'], 0).reshape(
                point_matrix.shape[0] - len(result_cluster['noise']), 3)
            part_label = np.delete(label_, result_cluster['noise'])
            print("删除噪点：K:", len(set(part_label)))
        elif -1 in result_cluster.keys():
            part_point_matrix = np.delete(point_matrix, result_cluster[-1], 0).reshape(
                point_matrix.shape[0] - len(result_cluster[-1]), 3)
            part_label = np.delete(label_, result_cluster[-1])
            print("删除噪点：K:", len(set(part_label)))
        else:
            part_point_matrix = point_matrix
            part_label = label_
    else:
        part_point_matrix, part_label = point_matrix, label_
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    try:
        print("轮廓系数（Silhouette Coefficient）越大越好", silhouette_score(part_point_matrix, part_label))
        print("CHS越大越好", calinski_harabasz_score(part_point_matrix, part_label))  # 类之间方差和类内方差
        print("DBS越小越好", davies_bouldin_score(part_point_matrix, part_label))  # DB值越小表示聚类结果同簇内部紧密，不同簇分离较远,类内距离越小，类间距离越大
        return [silhouette_score(part_point_matrix, part_label),
                calinski_harabasz_score(part_point_matrix, part_label),
                davies_bouldin_score(part_point_matrix, part_label)]
    except:
        print("K为1，无法评估")
        return None


def KANN(dist_matrix):
    dist_matrix_c = dist_matrix.copy()
    dist_matrix_c = np.sort(dist_matrix_c, axis=1)
    return np.mean(dist_matrix_c, axis=0)


def Compose(num_list, diff=False):
    # 返回num_list列表中不重复的两两组合
    # 例如[1,2,3] 返回[1,2], [1,3], [2,3]
    # diff表示组合两个同的组合
    import itertools
    if diff:  # 例如[[1,2],[3,4]] 返回[1,3], [1,4], [2,3],[2,4]
        return list(itertools.product(list(num_list[0]), list(num_list[1])))
    else:
        return list(itertools.combinations(list(num_list), 2))


def CHI(result_cluster, similar_matrix):
    '''
    Calinski-Harabasz Index被定义为簇间离散与簇内离散的比率，是通过评估类之间方差和类内方差来计算得分。该分值越大说明聚类效果越好
    :param result_cluster: key:簇中心id，value属于该簇的轨迹id
    :param similar_matrix: 相似度矩阵
    '''
    # center_diatance = []  # 记录簇内距离簇中心距离
    # for cluster_id in result_cluster.keys():
    #     for i in result_cluster[cluster_id]:
    #         center_diatance.append(np.linalg.norm(similar_matrix[cluster_id][i], 2))  # 添加每个数据点距离簇心的距离
    # print('平均簇心距离（越小越好）:', np.mean(center_diatance))
    N = similar_matrix.shape[0]  # 数据个数
    k = len(result_cluster)  # 聚类个数
    if k == 1:
        print("Calinski-Harabasz Index(越大越好):  ", np.nan)
    all_out_distance = []
    all_inner_distance = []
    import itertools
    all_track = list(itertools.chain.from_iterable(list(result_cluster.values())))
    for cluster_id in result_cluster.keys():
        inner_distance = []
        for tr_a, tr_b in Compose(result_cluster[cluster_id]):
            inner_distance.append(similar_matrix[tr_a, tr_b])
        all_out_distance.append(np.mean(inner_distance))
        all_inner_distance.extend(inner_distance)
    print("Calinski-Harabasz Index(越大越好):  ", np.std(all_out_distance)/np.std(all_inner_distance)*(N-k)/(k-1))
    return np.std(all_out_distance)/np.std(all_inner_distance)*(N-k)/(k-1)


def SC(result_cluster, similar_matrix=None, point_matrix=None):
    '''
    轮廓系数a：average(i向量到所有它属于的簇中其它点的距离)，b：min (i向量到某一不包含它的簇内的所有点的平均距离) S = (b-a) / max(a,b)平均，越大越好
    :param result_cluster: 聚类结果{cluster1:[tr1index,tr2index....],...}
    :param similar_matrix: 相似度矩阵
    :param point_matrix:没有计算相似度矩阵则需要从点来计算距离
    :param membership_mat:模糊聚类结果隶属度矩阵
    :param
    :return:
    '''
    if len(result_cluster) == 1:
        print("Silhouette Coefficient(越大越好):  ", np.nan)
    if similar_matrix is not None:
        n = similar_matrix.shape[0]
    else:
        n = point_matrix.shape[0]
    import itertools
    s = []  # s[i]
    all_track = list(itertools.chain.from_iterable(list(result_cluster.values())))  # 排除噪声
    all_cluster = list(result_cluster.keys())
    for cluster_id in result_cluster.keys():  # 遍历每个簇
        for track_id in result_cluster[cluster_id]:  # 遍历每个簇中的每个点
            a = similar_matrix[track_id, result_cluster[cluster_id]]  # 保存点到簇内距离的平均值
            b = []  # 保存每个点到其他簇的平均距离
            for other_cluster in all_cluster:
                if other_cluster == cluster_id:
                    continue
                b.append(np.mean(similar_matrix[track_id, result_cluster[other_cluster]]))
        s.append((np.min(b) - np.mean(a)) / max(np.mean(a), np.min(b)))
    print("Silhouette Coefficient(越大越好): ", np.mean(s))
    return np.mean(s)


def DVI(result_cluster, similar_matrix):
    '''
    Dunn Validity Index (邓恩指数)(DVI)计算任意两个簇元素的最短距离(类间)除以任意簇中的最大距离(类内)。 DVI越大意味着类间距离越大同时类内距离越小
    DVI = min(两个簇的最短距离（类间）)/max(簇内的最大距离，类内)
    :param result_cluster: 聚类结果{cluster1:[tr1index,tr2index....],...}
    :param similar_matrix: 相似度矩阵
    :return:
    '''
    if len(result_cluster) == 1:
        print("Dunn Validity Index(越大越好):  ", np.nan)
    other_cluster = []  # 记录距离其他簇的最小值
    innner_cluster = []  # 记录簇内的最大距离
    s = []
    for a, b in Compose(result_cluster.keys()):  # 遍历不同簇之间的最小距离
        for a_id, b_id in Compose([result_cluster[a], result_cluster[b]], diff=True):
            if len(similar_matrix.shape) == 3:
                s.append(np.linalg.norm(similar_matrix[a_id][b_id], 2))
            else:
                s.append(similar_matrix[a_id][b_id])
        other_cluster.append(min(s))
    for cluster_id in result_cluster.keys():  # 查找每个簇中的簇内最大距离
        b = []
        for a_id, b_id in Compose(result_cluster[cluster_id]):
            if len(similar_matrix.shape) == 3:
                b.append(np.linalg.norm(similar_matrix[a_id][b_id], 2))
            else:
                b.append(similar_matrix[a_id][b_id])
        if len(b) == 0:
            continue
        innner_cluster.append(max(b))
    print("Dunn Validity Index(越大越好):  ", np.min(other_cluster) / np.max(innner_cluster))
    return np.min(other_cluster) / np.max(innner_cluster)


def VCIM(result_cluster, point_matrix=None, membership_mat=None):
    '''
    越大越好
    :param result_cluster: 聚类结果
    :param point_matrix: 相似度矩阵
    :param membership_mat: 隶属度矩阵
    :return:
    '''
    N = len(result_cluster.keys())-1  # 簇个数(去噪声)
    # 1. 计算簇心
    CENTER = np.zeros((N, point_matrix.shape[1]))
    for cluster_id in result_cluster.keys():
        if cluster_id == -1:
            continue
        CENTER[cluster_id] = np.dot(point_matrix.T, membership_mat[:, cluster_id])
    # 2. 计算DBC
    DBC = 0  # 簇心间的距离
    for cluster_a, cluster_b in Compose(list(range(N))):
        DBC += np.linalg.norm(CENTER[cluster_a] - CENTER[cluster_b], 2)
    DBC = DBC * 2 / (N - 1) / N
    # 3. 计算DWC
    DWC = 0
    for cluster_id in result_cluster.keys():  # 遍历每个簇
        if cluster_id == -1:
            continue
        dwc = []  # 记录每个簇到簇心的距离
        for point_id in result_cluster[cluster_id]:  # 遍历簇中每个点
            dwc.append(np.linalg.norm(point_matrix[point_id] - CENTER[cluster_id], 2))
        DWC += np.mean(dwc)
    DWC = DWC / N
    # 4. 计算VCIM
    print("CVI :", DBC/DWC)
    return DBC/DWC


def Compactness(point_matrix, label_, cluster_centers):
    '''
    K=2计算聚类的紧密型：簇心到各个簇点之间的距离的平均值, 用来判断是否有聚类的必要
    :param result_cluster:
    :param similar_matrix:
    :return:簇心到各个簇点之间的距离的平均距离是否小于簇中心之间的距离
    '''
    distance = []  # 记录距离簇心的
    for i in range(2):
        for j in list(np.where(label_ == 0)[0]):
            distance.append(np.linalg.norm(point_matrix[j, :] - cluster_centers[i], 2))
    return np.mean(distance) < np.linalg.norm(cluster_centers[0]-cluster_centers[1], 2)


def CC(point_matrix):
    # 首先对数据进行归一化，存储为txt格式，用R语言进行一致性聚类得到K值
    point_matrix = np.array(point_matrix)
    np.save(os.path.join(OUTPUT_PATH, 'point_matrix.npy'), point_matrix)
    print("{} file saved over!".format(os.path.join(OUTPUT_PATH, 'point_matrix.npy')))
    # 用共识（一致性）聚类判断K值,rpy2库只能在linux中，win系统不支持
    # from rpy2 import robjects
    # from rpy2.robjects.packages import importr
    # ConsensusCluster = importr('ConsensusClusterPlus')
    # result = ConsensusCluster(point_matrix, maxK=6, reps=50, pItem=0.8, pFeature=1,
    #                      title="title", clusterAlg="hc", distance="pearson", plot="png")
    # print(result)


def Loss_Compress(before_compress, compression_index):
    '''
    计算轨迹压缩的质量（越小越好)
    :param before_compress: 压缩前的数据
    :param compression_index:压缩后的数据index
    :param after_compress: 压缩后经过处理后的数据的数据
    :return: 返回loss（越小越好）,compress（越大越好）, loss/compress(越小越好)
    '''
    compress = 1 - float(len(compression_index) / len(before_compress))
    loss = 0
    if type(before_compress) != list:
        before_compress = list(before_compress)
        compression_index = list(compression_index)
    after_compress = list(np.array(before_compress, dtype=np.float64)[compression_index])

    def cal_loss(point, start_point, end_point):
        # 计算point与start_point,end_point组成的线段在y轴上的差值
        if point[0] < start_point[0] or point[0] > end_point[0]:
            print("{} 点不在  {}：{}  范围内！！！".format(point, start_point, end_point))
            return None
        x = abs(end_point[0] - start_point[0])
        y = abs(end_point[1] - start_point[1])
        if end_point[1] - start_point[1] > 0:
            return abs((point[0] - start_point[0]) / x * y + start_point[1] - point[1])
        elif end_point[1] - start_point[1] == 0:
            return abs(end_point[1] - point[1])
        else:
            return abs(start_point[1] - (point[0] - start_point[0]) / x * y - point[1])

    for i, value in enumerate(before_compress):
        if i in compression_index:
            # pos = after_compress.index(value)
            pos = compression_index.index(i)
            if pos >= len(compression_index) - 1:
                break
            start = [i, value]  # 定位关键点
            end = [compression_index[pos+1], after_compress[pos+1]]
            continue
        loss += cal_loss([i, value], start, end)
    if compress == 0:
        return np.inf
    # print("loss : {};\ncompress: {};\nloss_compress(越小越好): {}".format(loss, compress, loss/compress))
    return loss/compress


def Compression_Accuracy(TR_a,TR_b, algorithm, nocomprssion_value):
    '''
    度量压缩轨迹准确度（越大越好）
    :param TR_a
    :param TR_b
    :param algorithm:测试的算法
    :param nocomprssion_value:没有经过压缩后的值
    :return: Accuracy rate（1-差值/原本正确的值） /压缩时间（压缩的时间+DTW比较的时间），较小的时间，有较高的准确率
    '''
    from Similarity.Similarity_Measurement_Methodology import ACDTW
    from Trajectory_Segmentation.Trajectory_Segmentation_Methodology import trajectory_segment
    import pandas as pd
    start = time.time()
    # 压缩轨迹
    feature_matrix = []
    feature_matrix.extend(trajectory_segment(TR_a, draw=False, algorithm=algorithm, cal_type='L', PCA_precess=False))
    feature_matrix.extend(trajectory_segment(TR_b, draw=False, algorithm=algorithm, cal_type='L', PCA_precess=False))
    feature_matrix = np.array(feature_matrix, dtype=np.float64)
    # 对特征进行归一化,统一量纲
    for i in range(2, feature_matrix.shape[1]):
        if (max(feature_matrix[:, i]) - min(feature_matrix[:, i])) != 0:
            feature_matrix[:, i] = (feature_matrix[:, i] - min(feature_matrix[:, i])) / (max(feature_matrix[:, i]) -
                                                                                         min(feature_matrix[:, i]))
    df = pd.DataFrame(data=feature_matrix, columns=['TrackId', 'LineId', 'amplitude_v', 'x_v', 'y_v',
                                                    'speed_v', 'radius_v', 'velocity_v', 'angle_v'])
    # 计算DTW
    comprssion_value = ACDTW(TR_a, TR_b, df, 'L', AC=False)
    end = time.time()
    Accuracy_rate = 1 - abs(comprssion_value - nocomprssion_value) / nocomprssion_value
    print("Accuracy rate : {}, \nTime: {},\nCompression_Accuracy:{}".format(
        Accuracy_rate, end - start, Accuracy_rate/(end - start)))


if __name__ == '__main__':
    # dist_matrix相似度矩阵已知，根据相似度矩阵来判断需要聚类最佳参数
    # file = os.path.join(OUTPUT_PATH, 'Similarity_matrix.txt')
    # dist_matrix = np.loadtxt(file, delimiter=',')
    # optics_parameters(dist_matrix, 0)
    # optics_parameters(dist_matrix, 1)
    center = [[2,2],[-2,-2],[0,0]]
    data, label = make_blobs(100,3,centers=center)
    print(data)
    CC(data)
    plt.figure()
    plt.scatter(data[:,0],data[:,1])
    plt.show()
    dist_matrix = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(i+1, data.shape[0]):
            dist_matrix[i][j] = dist_matrix[j][i] = np.linalg.norm(data[i]-data[j], 2)
    print(dist_matrix)
    optics_parameters(dist_matrix, 0)
