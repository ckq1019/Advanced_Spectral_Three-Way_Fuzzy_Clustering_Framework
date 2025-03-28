from Parameter_Selection_methodology import CHI
import numpy as np


def AP(TR_List, similar_matrix):
    '''
    用AP算法聚类
    :param TR_List:
    :param similar_matrix:
    :return:
    '''
    # 初始化R,A矩阵
    length = len(TR_List)
    A, R = np.zeros((length, length)), np.zeros((length, length))
    S = -similar_matrix  # 相似度矩阵需要处理下,设置参考值，并乘以-1
    # S[S == 0] = min(S.flat)  # 选用相似度最小值为参考值
    # S[S == 0] = np.median(S.flat)  # 选用相似度最小值为参考值
    S[S == 0] = -0.5
    lam = 0.5  # 阻尼系数,用于算法收敛

    def iter_update_R():
        '''
        计算吸引度矩阵，即R
        r(n+1)(i,k)=(1-λ)*r(n+1)(i,k)+λ*r(n)(i,k)
        r(n+1)(i,k)=S(i,k)-max(A(i,j)+R(i,j) j!=k) if i!=k
        r(n+1)(i,k)=S(i,k)-max(S(i,j) j!=k) if i==k
        '''
        # 更新R矩阵
        for i in range(length):
            for k in range(length):
                old_r = R[i][k]
                if i != k:
                    max1 = A[i][0] + R[i][0]  # 注意初始值的设置
                    for j in range(length):
                        if j != k:
                            if A[i][j] + R[i][j] > max1:
                                max1 = A[i][j] + R[i][j]
                    R[i][k] = S[i][k] - max1
                else:
                    max2 = S[i][0]  # 注意初始值的设置
                    for j in range(length):
                        if j != k:
                            if S[i][j] > max2:
                                max2 = S[i][j]
                    # 更新后的R[i][k]值
                    R[i][k] = S[i][k] - max2
                # 带入阻尼系数从新更新
                R[i][k] = (1 - lam) * R[i][k] + lam * old_r

    def iter_update_A():
        '''
        计算归属度矩阵，即A
        A(n+1)(i,k)=(1-λ)*A(n+1)(i,k)+λ*A(n)(i,k)
        '''
        for i in range(length):
            for k in range(length):
                old_a = A[i][k]
                if i == k:
                    max3 = R[0][k]  # 注意初始值的设置
                    for j in range(length):
                        if j != k:
                            if R[j][k] > 0:
                                max3 += R[j][k]
                            else:
                                max3 += 0
                    A[i][k] = max3
                else:
                    max4 = R[0][k]  # 注意初始值的设置
                    for j in range(length):

                        if j != k and j != i:
                            if R[j][k] > 0:
                                max4 += R[j][k]
                            else:
                                max4 += 0

                    if R[k][k] + max4 > 0:
                        A[i][k] = 0
                    else:
                        A[i][k] = R[k][k] + max4

                # 带入阻尼系数更新A值
                A[i][k] = (1 - lam) * A[i][k] + lam * old_a

    center = []
    compare = 0
    for i in range(1000):
        iter_update_R()
        iter_update_A()
        # 开始计算聚类中心
        for k in range(length):
            if R[k][k] + A[k][k] > 0:
                if k not in center:
                    center.append(k)
                else:
                    compare += 1
        if compare > 50:
            # 聚类中心不再变化
            break
    # 根据距离判断所属聚类中心
    result_cluster = {}
    for i in range(length):
        cluster = center[0]  # 假设属于第0个簇
        for k in center:
            if similar_matrix[i][k] < similar_matrix[i][cluster]:
                cluster = k
        if cluster not in result_cluster.keys():
            result_cluster[cluster] = []
        result_cluster[cluster].append(i)
    CHI(result_cluster, similar_matrix)
    return center
