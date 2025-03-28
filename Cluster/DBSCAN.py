import numpy as np
from Similarity.Similarity_Measurement_Methodology import ACDTW,TD


def expand_cluster(queue, TR_list, cluster_id, MinTRs, neighbors_mat):
    '''
    :param queue: 队列用来存储轨迹index
    :param TR_list:
    :param cluster_id:
    :param MinTRs:
    :param neighbors_mat: 邻接轨迹矩阵
    :return:返回每个簇的代表轨迹id
    '''
    while len(queue) != 0:
        track_index = queue.pop()
        TR_list[track_index].ClusterId = cluster_id
        TR_list[track_index].visited = True
        neighbors_index = np.nonzero(neighbors_mat[track_index])[0]
        for i in neighbors_index:
            if TR_list[i].ClusterId == -1 and i not in queue:
                queue.append(i)


def isNeighbors(TRi, TRj, eps, T_threshold=None, feature=None, type='P'):
    '''
    :param TRi: 轨迹1
    :param TRj: 轨迹2
    :param eps: 两个轨迹之间的距离阈值
    :param T_threshold: [0,1] 1无重叠时间，值越高重叠的越少,值越小重叠越高
    :return:
    '''
    if feature is not None:
        feature = feature[(feature['TrackId'] == float(TRi.TrackId)) | (feature['TrackId'] == float(TRj.TrackId))]
    # 多因子豪斯多夫
    # simlar = MFHD(TRi, TRj, type)
    # 多因子线段DTW
    if T_threshold != None:
        if TD(TRi, TRj) <= T_threshold and ACDTW(TRi, TRj, feature, type) <= eps:
            return True
    else:
        if ACDTW(TRi, TRj, feature, type) <= eps:
            return True
    return False


def DBSCAN(TR_List, eps, MinTRs, T_threshold=0.5, similar_matrix=None):
    '''
    :param TR_List: 轨迹列表
    :param eps: 相似度阈值
    :param MinTRs: 最小线段数字
    :param T_threshold: 时间阈值
    :param u_mat: 各簇的隶属徐
    :return:
    '''
    # 得到邻接轨迹列表（上三角矩阵）,如果相邻的话就为1
    neighbors_mat = similar_matrix.copy()
    for i in range(len(TR_List)):
        TR_List[i].visited = False  # 初始化轨迹访问
        TR_List[i].ClusterId = -1
        for j in range(i+1, len(TR_List)):
            if neighbors_mat[i][j] <= eps:
                neighbors_mat[i][j] = 1
                neighbors_mat[j][i] = 1
            else:
                neighbors_mat[i][j] = 0
                neighbors_mat[j][i] = 0
    cluster_id = 0
    for TR_index in range(len(TR_List)):
        if not TR_List[TR_index].visited:
            TR_List[TR_index].visited = True
            neighbors_index = np.nonzero(neighbors_mat[TR_index])
            if len(neighbors_index[0]) >= MinTRs:
                # 核心线段
                queue = [TR_index]  # 初始化队列,队列中装是的index
                expand_cluster(queue, TR_List, cluster_id, MinTRs, neighbors_mat)
                cluster_id += 1
    result_dict = {}
    for TR_index in range(len(TR_List)):
        if TR_List[i].ClusterId != -1:
            if TR_List[i].ClusterId not in result_dict.keys():
                result_dict[TR_List[i].ClusterId] = []
            result_dict[TR_List[i].ClusterId].append(i)
    return result_dict


def OPTICS(TR_List, eps, MinTRs, similar_matrix, T_threshold=None):
    '''
    OPTICS算法
    :param TR_List: 轨迹列表
    :param eps: 距离（相似度）阈值
    :param MinTRs: 最小轨迹数
    :param T_threshold: 时间阈值
    :param similar_matrix: 相似度矩阵
    :return:
    '''
    dist_matrix = similar_matrix.copy()
    TR_queue = [0]
    cluster = 1
    reslut = {}
    while len(np.nonzero(dist_matrix)[0]):
        TR_index = TR_queue[-1]
        for min_index in np.argsort(dist_matrix[TR_index]):
            if min_index in np.nonzero(dist_matrix[TR_index])[0]:
                break
        if dist_matrix[min_index][TR_index] <= eps:
            dist_matrix[TR_index].fill(0)
            dist_matrix[:, TR_index].fill(0)
            TR_queue.append(min_index)
        else:
            dist_matrix[TR_index].fill(0)
            dist_matrix[:, TR_index].fill(0)
            if len(TR_queue) >= MinTRs:
                # 存储聚类，队列中为一个簇
                cluster_queue = []
                while len(TR_queue):
                    tr_index = TR_queue.pop()
                    TR_List[tr_index].ClusterId = cluster
                    cluster_queue.append(tr_index)
                reslut[cluster] = cluster_queue
                cluster += 1
            else:
                TR_queue.clear()
            TR_queue.append(min_index)
    return reslut

# def OPTICS(TR_List, eps, MinTRs, similar_matrix, T_threshold=None):
#     '''
#     OPTICS算法
#     :param TR_List: 轨迹列表
#     :param eps: 距离（相似度）阈值
#     :param MinTRs: 最小轨迹数
#     :param T_threshold: 时间阈值
#     :param similar_matrix: 相似度矩阵
#     :return:
#     '''
#     next_id = 0
#     cluster_id = 1
#     result_cluster = {}
#     for i, tr in enumerate(TR_List):
#         if not tr.visited:
#             TR_queue = [next_id]
#             Queue = [next_id]  # 存放一个簇中的轨迹id
#             dis_Neighbors = 0  # 记录簇中心
#             dis_cen = 0  # 记录簇中心点的与周边点的距离
#             while len(TR_queue) != 0:
#                 TR_index = TR_queue.pop()
#                 TR_List[TR_index].visited = True
#                 for sort_index in np.argsort(similar_matrix[TR_index]):
#                     if similar_matrix[sort_index][TR_index] <= eps and similar_matrix[sort_index][TR_index] > 0:
#                         if sort_index not in Queue:
#                             TR_queue.append(sort_index)
#                             Queue.append(sort_index)
#                         dis_Neighbors += similar_matrix[sort_index][TR_index]
#                     elif similar_matrix[sort_index][TR_index] > eps:
#                         next_id = sort_index
#                         break
#                 if dis_cen < dis_Neighbors:
#                         dis_cen = dis_Neighbors
#                         cen_id = TR_index
#             if len(Queue) >= MinTRs:
#                 result_cluster[cen_id] = Queue
#                 cluster_id += 1
#     # while len(np.nonzero(dist_matrix)[0]):
#     #
#     #     while len(TR_queue) != 0:
#     #         TR_index = TR_queue.pop()
#     #         Queue.append(TR_index)
#     #         dis_Neighbors = 0
#     #         for min_index in np.argsort(dist_matrix[TR_index]):
#     #             if min_index in np.nonzero(dist_matrix[TR_index])[0] and dist_matrix[min_index][TR_index] < eps:
#     #                 if min_index not in TR_queue:
#     #                     TR_queue.append(min_index)
#     #                     Queue.append(min_index)
#     #
#     #             elif min_index in np.nonzero(dist_matrix[TR_index])[0]:
#     #                 next_id = min_index  # 记录大于eps的最小距离轨迹的id
#     #                 break
#     #         dist_matrix[TR_index].fill(0)
#     #         dist_matrix[:, TR_index].fill(0)
#     #
#     #     Queue = list(set(Queue))
#     #     if len(set(Queue)) >= MinTRs:
#     #         result_cluster[cen_id] = Queue
#     #     TR_queue.append(next_id)
#     CH(result_cluster, similar_matrix)
#     return result_cluster
