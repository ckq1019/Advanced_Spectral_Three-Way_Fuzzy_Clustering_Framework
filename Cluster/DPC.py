import numpy as np
import matplotlib.pyplot as plt
percent = 2.0 / 100
noise = [19,64,231,142,45,172]
K=5


def get_distance_matrix(datas):
    n = np.shape(datas)[0]
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # v_i = datas[i, :]
            # v_j = datas[j, :]
            # distance_matrix[i, j] = np.sqrt(np.dot((v_i - v_j), (v_i - v_j)))
            distance_matrix[i, j] = np.linalg.norm(datas[i] - datas[j], 2)
    return distance_matrix


def select_dc(distance_matrix):
    n = np.shape(distance_matrix)[0]
    distance_array = np.reshape(distance_matrix, n * n)
    position = int(n * (n - 1) * percent)
    dc = np.sort(distance_array)[position + n]
    return dc


def get_local_density(distance_matrix, dc, method=None):
    n = np.shape(distance_matrix)[0]
    rhos = np.zeros(n)
    for i in range(n):
        if method is None:
            # rhos[i] = np.where(distance_matrix[i, :] < dc)[0].shape[0] - 1
            for j in range(n):
                if j == i:
                    continue
                rhos[i] += np.exp(-(distance_matrix[i, j]/dc)**2)
        else:
            pass
    return rhos


def get_deltas(distance_matrix, rhos):
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


def find_k_centers(rhos, deltas, k):
    rho_and_delta = rhos * deltas
    centers = np.argsort(-rho_and_delta)
    return centers[:k]


def density_peal_cluster(rhos, centers, nearest_neighbor):
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
    for i in noise:
        labels[i] = -1
    return labels


def draw_decision(datas, rhos, deltas):
    n = np.shape(datas)[0]
    for i in range(n):
        plt.scatter(rhos[i], deltas[i], s=16, color=(0, 0, 0))
        plt.annotate(str(i), xy=(rhos[i], deltas[i]), xytext=(rhos[i], deltas[i]))
        plt.xlabel('local density-ρ')
        plt.ylabel('minimum distance to higher density points-δ')
    plt.show()


def main():
    import os
    import numpy as np
    datas = np.load(os.path.join(r'D:\Trajectory_analysis\Dimension_Reduction\output',  'all_point_matrix.npy'))
    distance_matrix = get_distance_matrix(datas)  # 计算距离矩阵
    dc = select_dc(distance_matrix)  # 确定邻域截断距离dc
    rhos = get_local_density(distance_matrix, dc)  # 计算局部密度和相对距离
    deltas, nearest_neighbor = get_deltas(distance_matrix, rhos)
    draw_decision(datas, rhos, deltas)  # 绘制决策图，选取聚类中心
    centers = find_k_centers(rhos, deltas, K)
    print(centers)
    a = density_peal_cluster(rhos, centers, nearest_neighbor)
    print("DPC:", list(a))

    fig = plt.figure()
    # 构建xyz
    ax = plt.axes(projection="3d")
    ax.scatter3D(datas[np.where(a == -1)[0], 0], datas[np.where(a == -1)[0], 1], datas[np.where(a == -1)[0], 2],
                 color="grey")
    color_bar = ["red", "blue", "green", 'yellow','pink','purple','black','cyan','brown']
    for i in range(K):
        ax.scatter3D(datas[centers[i], 0], datas[centers[i], 1], datas[centers[i], 2], marker='^')
        if i in centers:
            continue
        ax.scatter3D(datas[np.where(a==i)[0], 0], datas[np.where(a==i)[0], 1], datas[np.where(a==i)[0], 2], color=color_bar[i])

    # ax.scatter3D(x, y, z, color="red")
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    plt.show()

    from sklearn.cluster import OPTICS
    optics = OPTICS(min_samples=10).fit(datas)
    fig = plt.figure()
    a = optics.labels_
    print("OPTICS:", list(a))
    # 构建xyz
    ax = plt.axes(projection="3d")
    ax.scatter3D(datas[np.where(a == -1)[0], 0], datas[np.where(a == -1)[0], 1], datas[np.where(a == -1)[0], 2],
                 color="grey")
    for i in range(K):
        ax.scatter3D(datas[np.where(a == i)[0], 0], datas[np.where(a == i)[0], 1], datas[np.where(a == i)[0], 2],
                     color=color_bar[i])
    # ax.scatter3D(x, y, z, color="red")
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    plt.show()

    from sklearn.cluster import SpectralClustering
    optics = SpectralClustering(n_clusters=K, gamma=1.1).fit(datas)
    fig = plt.figure()
    a = optics.labels_
    # 构建xyz
    ax = plt.axes(projection="3d")
    ax.scatter3D(datas[np.where(a == -1)[0], 0], datas[np.where(a == -1)[0], 1], datas[np.where(a == -1)[0], 2],
                 color="grey")
    for i in range(K):
        ax.scatter3D(datas[np.where(a == i)[0], 0], datas[np.where(a == i)[0], 1], datas[np.where(a == i)[0], 2],
                     color=color_bar[i])
    # ax.scatter3D(x, y, z, color="red")
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    plt.show()
    print("SpectralClustering:", list(a))


if __name__ == '__main__':
    main()