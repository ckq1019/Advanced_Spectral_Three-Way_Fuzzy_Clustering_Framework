from Global_Variable import *
import numpy as np
from pyecharts.charts import Line
import pyecharts.options as opts


def SOM():
    return None


class DPC:
    def __init__(self, percent=0.02):
        self.percent = percent
        self.kernel = 'Gaussian'  # 选择计算核方法
        self.n_clusters = 0
        self.noise = []
        self.rhos = None
        self.deltas = None
        self.nearest_neighbor = None
        self.labels_ = []

    def fit(self, point_matrix):
        datas = point_matrix  # 数据集
        distance_matrix = self.get_distance_matrix(datas)  # 计算距离矩阵
        dc = self.select_dc(distance_matrix)  # 确定邻域截断距离dc
        rhos = self.get_local_density(distance_matrix, dc, 'Gaussian')  # 计算局部密度和相对距离
        deltas, nearest_neighbor = self.get_deltas(distance_matrix, rhos)
        # self.draw_decision(datas, rhos, deltas)
        self.rhos = rhos
        self.deltas = deltas
        self.nearest_neighbor = nearest_neighbor

    def cluster(self, n_clusters, noise=[]):
        self.n_clusters = n_clusters  # 聚类个数
        self.noise = noise
        centers = self.find_k_centers(self.rhos, self.deltas, self.n_clusters)
        self.labels_ = self.density_peal_cluster(self.rhos, centers, self.nearest_neighbor)
        return self.labels_

    def get_distance_matrix(self, datas):
        n = np.shape(datas)[0]
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distance_matrix[i, j] = np.linalg.norm(datas[i] - datas[j], 2)
        return distance_matrix

    def select_dc(self, distance_matrix):
        n = np.shape(distance_matrix)[0]
        distance_array = np.reshape(distance_matrix, n * n)
        position = int(n * (n - 1) * self.percent)
        dc = np.sort(distance_array)[position + n]
        return dc

    def get_local_density(self, distance_matrix, dc, method=None):
        n = np.shape(distance_matrix)[0]
        rhos = np.zeros(n)
        for i in range(n):
            if method is None:
                rhos[i] = np.where(distance_matrix[i, :] < dc)[0].shape[0] - 1
            elif method == 'Gaussian':
                for j in range(n):
                    if j == i:
                        continue
                    rhos[i] += np.exp(-(distance_matrix[i, j] / dc) ** 2)
            else:
                pass
        return rhos

    def get_deltas(self, distance_matrix, rhos):
        n = np.shape(distance_matrix)[0]
        deltas = np.zeros(n)
        nearest_neighbor = np.zeros(n)
        rhos_index = np.argsort(-rhos)
        for i, index in enumerate(rhos_index):
            if i == 0:
                continue
            higher_rhos_index = rhos_index[:i]
            deltas[index] = np.min(distance_matrix[index, higher_rhos_index])
            nearest_neighbors_index = np.argmin(distance_matrix[index, higher_rhos_index])
            nearest_neighbor[index] = higher_rhos_index[nearest_neighbors_index].astype(int)
        deltas[rhos_index[0]] = np.max(deltas)
        return deltas, nearest_neighbor

    def draw_decision(self, datas, rhos, deltas):  # 绘制决策图，选取聚类中心
        n = np.shape(datas)[0]
        import matplotlib.pyplot as plt
        plt.figure()
        for i in range(n):
            plt.scatter(rhos[i], deltas[i], s=16, color=(0, 0, 0))
            plt.annotate(str(i), xy=(rhos[i], deltas[i]), xytext=(rhos[i], deltas[i]))
            plt.xlabel('local density-ρ')
            plt.ylabel('minimum distance to higher density points-δ')
        plt.show()
        plt.close()

    def find_k_centers(self, rhos, deltas, k):
        rho_and_delta = rhos * deltas
        centers = np.argsort(-rho_and_delta)
        return centers[:k]

    def density_peal_cluster(self, rhos, centers, nearest_neighbor):
        k = np.shape(centers)[0]
        if k == 0:
            print("Can't find any center")
            return
        n = np.shape(rhos)[0]
        labels = -1 * np.ones(n).astype(int)

        for i, center in enumerate(centers):
            labels[center] = i

        rhos_index = np.argsort(-rhos)
        for i, index in enumerate(rhos_index):
            if labels[index] == -1:
                labels[index] = labels[int(nearest_neighbor[index])]
        for i in self.noise:
            labels[i] = -1
        return labels


def consolidation(label, center=None, cluster_dic=None):
    '''
    根据label得到对应的dic
    :param label:得到的聚类结果
    :param center:簇中心，如果是None的就默认数字（0，1，2，3，4）
    :return:{-1:[], 1:[tr_index...],...} or {center_id:[track_id,...]}
    '''

    def find_center(index_list, center_list):
        for i in index_list:
            if i in center_list:
                return i
        return -1

    if label is not None:
        label = np.array(label)
        result_cluster = {-1: [], }
        # delete = []  # 删除少于5个轨迹的cluster_id
        for i in set(label):
            if i == -1:
                result_cluster[-1] = list(np.where(label == i)[0])
                continue
            if center is not None:
                cen = find_center(list(np.where(label == i)[0]), center)  # 找到该簇的簇中心
            else:
                cen = i
            result_cluster[cen] = list(np.where(label == i)[0])
            # if len(result_cluster[cen]) < 5:  # 如果聚类数小于5则归为噪声
            #     delete.append(cen)
        # for i in delete:
        #     result_cluster[-1].extend(result_cluster[i])
        #     result_cluster.pop(i)
        # result_cluster.pop(-1)
        # if -1 in result_cluster.keys():
        #     result_cluster.pop(-1)  # 删除噪点聚类
        return result_cluster
    else:
        delete = []
        cluster_dic[-1] = []
        for i in cluster_dic.keys():
            if len(cluster_dic[i]) < 5:
                delete.append(i)
        for i in delete:
            cluster_dic[-1].extend(cluster_dic[i])
            cluster_dic.pop(i)
        return cluster_dic


def Draw(datas, label_):
    import matplotlib.pyplot as plt
    plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(datas[np.where(label_ == -1)[0], 0], datas[np.where(label_ == -1)[0], 1],
                 datas[np.where(label_ == -1)[0], 2], color="grey")  # 噪点
    color_bar = ["red", "blue", "green", 'yellow', 'pink', 'purple']
    for i in range(len(set(label_))):
        ax.scatter3D(datas[np.where(label_ == i)[0], 0], datas[np.where(label_ == i)[0], 1],
                     datas[np.where(label_ == i)[0], 2], color=color_bar[i])
    # [轨迹变化相似度， 轨迹出发结束位置点，时间戳距离]
    ax.set(xlabel="Change trend similarity", ylabel="First and last similarity", zlabel="Time similarity")
    plt.show()
    plt.close()


def Parameter_Selection_Line(parameter, file=None):
    parameter = np.array(parameter, dtype=np.float64)
    line = Line(opts.InitOpts())
    name = ['SC', 'CHS', 'DBS']
    line.add_xaxis([str(i) for i in parameter[:, 0].tolist()])
    # line.extend_axis(yaxis=opts.AxisOpts(type_="value"), name="CHS")
    for i in range(1, parameter.shape[1]):
        line.add_yaxis(series_name=name[i-1], y_axis=parameter[:, i].tolist(), label_opts=opts.LabelOpts(is_show=False))
    line.set_global_opts(toolbox_opts=opts.ToolboxOpts(is_show=True),
                         xaxis_opts=opts.AxisOpts(name="K", type_='category'),
                         yaxis_opts=opts.AxisOpts(type_='value', axisline_opts=opts.AxisLineOpts(is_show=True),
                                                  splitline_opts=opts.SplitLineOpts(is_show=True)))
    if file is None:
        file = os.path.join(PARA_SELECTION_PATH, 'Parameter_Selection.html')
    line.render(file)
    print("{} file saved!".format(file))


def ClusterModel(model_name, point_matrix, K=0, similar_matrix=None, draw=False, save=False, denoise=False):
    '''
    聚类模型
    :param model_name: 选择需要聚类的算法名称
    :param K: 算则聚类K值
    :param point_matrix:聚类数据
    :param draw:是否化结果图
    :param save:对聚类结果进行保存
    :param denoise:是否进行降噪，只适用于SC
    :return: 返回聚类结果
    '''
    from Cluster.Parameter_Selection_methodology import Evaluation_Model
    if type(model_name) == str:  # 单个模型聚类，如果有结果的话直接读结果
        file_path = os.path.join(CLUSTER_PATH, "cluster_point_matrix.npy")
        if os.path.exists(file_path):
            print("{} file read result of cluster!".format(file_path))
            label_ = np.load(file_path)
            return consolidation(label_[:, 3])
        model_name = [model_name]
    min_k = int(min(50, point_matrix.shape[0] / 2))
    # # 判断是否适合聚类（利用Kmeans算法）
    # from sklearn.cluster import KMeans
    # kmeans = KMeans(n_clusters=2).fit(point_matrix)
    # if not Compactness(point_matrix, kmeans.labels_, kmeans.cluster_centers_):  # 判断是否需要聚类
    #     # 如果簇心距离大于簇间距离，则无需聚类
    #     print("无需聚类")
    #     return None
    for model in model_name:
        print("{}模型评估：".format(model))
        if model == 'AP':
            from sklearn.cluster import AffinityPropagation
            ap = AffinityPropagation().fit(point_matrix)
            if draw:
                Draw(point_matrix, ap.labels_)
            result_cluster = consolidation(ap.labels_)
            Evaluation_Model(result_cluster, similar_matrix, point_matrix, ap.labels_)
        elif model == 'KMeans':
            from sklearn.cluster import KMeans
            if K:
                kmeans = KMeans(n_clusters=K).fit(point_matrix)
                if draw:
                    Draw(point_matrix, kmeans.labels_)
                result_cluster = consolidation(kmeans.labels_)
                Evaluation_Model(result_cluster, similar_matrix, point_matrix, kmeans.labels_)
            else:
                for k in range(2, min_k):
                    kmeans = KMeans(n_clusters=k).fit(point_matrix)
                    if draw:
                        Draw(point_matrix, kmeans.labels_)
                    result_cluster = consolidation(kmeans.labels_)
                    Evaluation_Model(result_cluster, similar_matrix, point_matrix, kmeans.labels_)
        elif model == 'SC':
            label_ = np.zeros((point_matrix.shape[0], ))
            if denoise:
                from sklearn.neighbors import LocalOutlierFactor
                label_ = LocalOutlierFactor(n_neighbors=5).fit_predict(point_matrix)
            from sklearn.cluster import SpectralClustering
            if K:
                sc = SpectralClustering(n_clusters=K, gamma=1.3).fit(point_matrix[label_ != -1, :])
                label_[label_ != -1] = sc.labels_
                if draw:
                    Draw(point_matrix, label_)
                result_cluster = consolidation(label_)
                Evaluation_Model(result_cluster, similar_matrix, point_matrix, label_, except_noise=True)
            else:
                if draw:
                    parameter = []  # 添加参数
                for k in range(2, min_k):
                    sc = SpectralClustering(n_clusters=k, gamma=1.3).fit(point_matrix[label_ != -1, :])
                    label_[label_ != -1] = sc.labels_
                    result_cluster = consolidation(label_)
                    if draw:
                        parameter.append([k]+Evaluation_Model(result_cluster, similar_matrix,  point_matrix, label_,
                                                          except_noise=True))
                    else:
                        Evaluation_Model(result_cluster, similar_matrix, point_matrix, label_, except_noise=True)
                if draw:
                    Parameter_Selection_Line(parameter)
        elif model == 'OPTICS':
            from sklearn.cluster import OPTICS
            optics = OPTICS(min_samples=10).fit(point_matrix)
            if draw:
                Draw(point_matrix, optics.labels_)
            result_cluster = consolidation(optics.labels_)
            Evaluation_Model(result_cluster, similar_matrix, point_matrix, optics.labels_)
        elif model == 'DPC':
            dpc = DPC()
            dpc.fit(point_matrix)
            dpc.cluster(n_clusters=K, noise=[114,66,146,67])
            if draw:
                Draw(point_matrix, dpc.labels_)
            result_cluster = consolidation(dpc.labels_)
            Evaluation_Model(result_cluster, similar_matrix, point_matrix, dpc.labels_)
        elif model == 'FCM':
            from Cluster.FCM import FCM
            if K:
                cluster_dic = FCM(point_matrix, K)
                result_cluster = consolidation(label=None, center=None, cluster_dic=cluster_dic)
                Evaluation_Model(result_cluster, similar_matrix, point_matrix, None)
            else:
                for k in range(2, min_k):
                    cluster_dic = FCM(point_matrix, k)
                    result_cluster = consolidation(label=None, center=None, cluster_dic=cluster_dic)
                    Evaluation_Model(result_cluster, similar_matrix, point_matrix, None)
        elif model == 'HC':
            from sklearn.cluster import AgglomerativeClustering
            if K:
                ac = AgglomerativeClustering(n_clusters=K).fit(point_matrix)
                if draw:
                    Draw(point_matrix, ac.labels_)
                result_cluster = consolidation(ac.labels_)
                Evaluation_Model(result_cluster, similar_matrix, point_matrix, ac.labels_)
            else:
                for k in range(2, min_k):
                    ac = AgglomerativeClustering(n_clusters=k).fit(point_matrix)
                    if draw:
                        Draw(point_matrix, ac.labels_)
                    result_cluster = consolidation(ac.labels_)
                    Evaluation_Model(result_cluster, similar_matrix, point_matrix, ac.labels_)
        print("{} over!".format(model))
        print("-" * 16)
        if save:
            cluster_point_matrix = np.zeros((point_matrix.shape[0], point_matrix.shape[1]+1))
            cluster_point_matrix[:, :3] = point_matrix
            for i in result_cluster.keys():
                cluster_point_matrix[result_cluster[i], 3] = i
            file_path = os.path.join(CLUSTER_PATH, "cluster_point_matrix.npy")
            np.save(file_path, cluster_point_matrix)
            print("{} file saved!".format(file_path))
    return result_cluster


if __name__ == '__main__':
    # 各硬聚类模型比较
    similar_matrix = np.load(os.path.join(r"D:\Trajectory_analysis\Similarity\output", 'Similarity_matrix.npy'))
    all_similar_matrix = (similar_matrix[:, :, 0] ** 2 + similar_matrix[:, :, 1] ** 2 + similar_matrix[:, :, 2] ** 2) ** 1/2
    from sklearn.manifold import MDS
    point = MDS(n_components=2, dissimilarity="precomputed").fit_transform(all_similar_matrix)

    def SC(labels):
        s = []
        all_cluster = list(set(labels))  # 所有聚类(不包括噪声)
        if -1 in all_cluster:
            all_cluster.remove(-1)
        if len(all_cluster) <= 1:
            return -1
        for cluster_id in all_cluster:  # 遍历每个簇
            if len(np.where(labels == cluster_id)[0]) == 1:  # 该簇只有一个成员
                continue
            for track_id in np.where(labels == cluster_id)[0]:  # 遍历每个簇中的每个数据
                # 保存点到簇内距离的平均值
                dis = all_similar_matrix[track_id, labels == cluster_id]
                a = dis[dis != 0]  # 排除自己到自己的距离
                # if len(a) == 0:
                #     continue
                a = np.mean(a)
                b = []  # 存放其他簇的平均距离
                for other_cluster in all_cluster:
                    if other_cluster == cluster_id:
                        continue
                    dis = all_similar_matrix[track_id, labels == other_cluster]
                    # if sum(dis) == 0:
                    #     continue
                    b.append(np.mean(dis[dis != 0]))
                s.append((np.min(b) - a) / max(a, np.min(b)))
        return np.nanmean(s)

    def SC_part(labels, j):
        s = []
        all_cluster = list(set(labels))  # 所有聚类(不包括噪声)
        if -1 in all_cluster:
            all_cluster.remove(-1)
        if len(all_cluster) <= 1:
            return -1
        for cluster_id in all_cluster:  # 遍历每个簇
            if len(np.where(labels == cluster_id)[0]) == 1:  # 该簇只有一个成员
                continue
            for track_id in np.where(labels == cluster_id)[0]:  # 遍历每个簇中的每个数据
                # 保存点到簇内距离的平均值
                dis = similar_matrix[track_id, labels == cluster_id, j]
                a = dis[dis != 0]  # 排除自己到自己的距离
                # if len(a) == 0:
                #     continue
                a = np.mean(a)
                b = []  # 存放其他簇的平均距离
                for other_cluster in all_cluster:
                    if other_cluster == cluster_id:
                        continue
                    dis = similar_matrix[track_id, labels == other_cluster, j]
                    # if sum(dis) == 0:
                    #     continue
                    b.append(np.mean(dis[dis != 0]))
                s.append((np.min(b) - a) / max(a, np.min(b)))
        return np.nanmean(s)


    alg = ["AP", "Kmeans", "SC", "HC", "OPTICS", "DPC"]
    for i in alg:
        if i == "AP":
            from sklearn.cluster import AffinityPropagation
            ap = AffinityPropagation().fit(point)
            labels = ap.labels_
        if i == "Kmeans":
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=4).fit(point)
            labels = kmeans.labels_
        if i == "SC":
            from sklearn.cluster import SpectralClustering
            sc = SpectralClustering(n_clusters=4).fit(point)
            labels = sc.labels_
        if i == "HC":
            from sklearn.cluster import AgglomerativeClustering
            hc = AgglomerativeClustering(n_clusters=4).fit(point)
            labels = hc.labels_
        if i == "OPTICS":
            from sklearn.cluster import OPTICS
            optics = OPTICS(min_samples=5).fit(point)
            labels = optics.labels_
        if i == "DPC":
            dpc = DPC()
            dpc.fit(point)
            dpc.cluster(n_clusters=4, noise=[])
            labels = dpc.labels_
        for j in range(3):
            print("SC_{} of {}:".format(j, i), SC_part(labels, j))
        print("SC of {}:".format(i), SC(labels))
        print("-" * 15)
