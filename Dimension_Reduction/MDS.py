import numpy as np


class MDS:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, similar_matrix):
        # MDS的思路就是保持新空间与原空间的相对位置关系，先用原空间的距离矩阵D，求得新空间的内积矩阵B，再由内积矩阵B求得新空间的表示方法Z
        # https://blog.csdn.net/csdn_inside/article/details/86004733
        m = similar_matrix.shape[0]
        dist = np.array(similar_matrix)
        disti = np.zeros(m)
        distj = np.zeros(m)
        B = np.zeros((m, m))
        for i in range(m):
            disti[i] = np.mean(dist[i, :])
            distj[i] = np.mean(dist[:, i])
        distij = np.mean(dist)
        for i in range(m):
            for j in range(m):
                B[i, j] = -0.5 * (dist[i, j] - disti[i] - distj[j] + distij)
        lamda, V = np.linalg.eigh(B)
        index = np.argsort(-lamda)[:self.n_components]
        diag_lamda = np.sqrt(np.diag(-np.sort(-lamda)[:self.n_components]))
        V_selected = V[:, index]
        Z = V_selected.dot(diag_lamda)
        return Z
