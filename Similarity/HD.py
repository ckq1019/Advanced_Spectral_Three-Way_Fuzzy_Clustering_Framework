import numpy as np


def MFHD(TR_a, TR_b, feature=None, cal_type='P'):
    '''
    计算TR_a和TR_b之间的Hausdorff距离
    :param TR_a:
    :param TR_b:
    :param feature: 处理（压缩）后的数据
    :param cal_type: 计算对象
    :return:
    '''
    if feature is not None:
        W = None
        for i in feature.columns:
            if 'w:' in i:
                if W is None:
                    W = []
                W.append(np.float(i[i.find(':')+1:]))
    if feature is None:
        # TR_a,TR_b是list
        TR_a = np.array(TR_a, dtype=np.float64)
        TR_b = np.array(TR_b, dtype=np.float64)
        n, m = TR_a.shape[0], TR_b.shape[0]
    else:
        n, m = len(feature[feature['TrackId'] == float(TR_a.TrackId)].values), len(
            feature[feature['TrackId'] == float(TR_b.TrackId)].values)
    dist_matrix = np.zeros((n+1, m+1))
    from Similarity.Similarity_Measurement_Methodology import Multifactor_distance
    for i in range(n):
        for j in range(m):
            if cal_type == 'P':
                dist_matrix[i][j] = Multifactor_distance(list(feature[feature['TrackId'] == float(
                    TR_a.TrackId)].values)[i], list(feature[feature['TrackId'] == float(TR_b.TrackId)].values)[j], W=W)
            elif cal_type == 'L':
                dist_matrix[i][j] = Multifactor_distance(list(feature[(feature['TrackId'] == float(TR_a.TrackId)) &
                                                        (feature['LineId'] == float(i))].values[0]),
                                            list(feature[(feature['TrackId'] == float(TR_b.TrackId)) &
                                                        (feature['LineId'] == float(j))].values[0]), W=W)
                if TR_a.line_list[i].weight != 0 and TR_b.line_list[j].weight != 0:
                    dist_matrix[i][j] *= (TR_a.line_list[i].weight + TR_b.line_list[j].weight)
            elif cal_type == 'Point':
                dist_matrix[i][j] = Multifactor_distance(TR_a[i], TR_b[j], cal_type=cal_type)
            if dist_matrix[n][j] == 0:
                dist_matrix[n][j] = dist_matrix[i][j]
            elif dist_matrix[n][j] > dist_matrix[i][j]:
                dist_matrix[n][j] = dist_matrix[i][j]
        dist_matrix[i][j+1] = np.min(dist_matrix[i][:-1])
    dist_matrix[i + 1][j] = np.max(dist_matrix[i + 1][:-1])
    print('MFHD距离相似度： ', max(dist_matrix[:, -1]))
    return max(dist_matrix[:, -1])

# def mfdist(Pa, Pb):
#     if Pa == Pb:
#         return 0
#     if Pa.velocity == 0 or Pb.velocity == 0:
#         return 0
#     W_dist = 0.4
#     W_velocity = 0.3
#     W_angle = 0.3
#     W_amplitude = 0
#     W_speed_average = 0
#     W_effective_radius = 0
#     pos_a = Pa.get_to_array()[3:5]
#     pos_b = Pb.get_to_array()[3:5]
#     return W_dist*dist(pos_a, pos_b)+W_velocity*abs(Pa.velocity-Pb.velocity)+W_angle*abs(Pa.angle-Pb.angle) + \
#            W_amplitude*abs(Pa.amplitude-Pb.amplitude)+W_speed_average*abs(Pa.speed_average-Pb.speed_average) + \
#            W_effective_radius*abs(Pa.effective_radius-Pb.effective_radius)

