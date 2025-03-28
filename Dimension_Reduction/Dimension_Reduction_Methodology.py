import numpy as np
from Global_Variable import *


class SimilarMatrixProjection:
    # 根据相似度矩阵映射到低维数据
    def __init__(self, algorithm, n_components=1, reshuffle=False, filepath=os.path.join(POINT_MATRIX, "all_point_matrix.npy")):
        self.algorithm = algorithm  # 选择的降维算法:TSNE, ISOMAP, MDS, UMAP，其他待添加
        self.n_components = n_components  # 降维后的数据维度
        self.reshuffle = reshuffle  # 是否重新生成数据
        self.file_path = filepath

    def fit(self, similar_matrix):
        if self.reshuffle or not os.path.exists(self.file_path):
            print(" Use {} updating file ".format(self.algorithm))
            point_matrix = np.zeros((similar_matrix.shape[0], similar_matrix.shape[-1] * self.n_components))
            if self.algorithm == 'TSNE':
                from sklearn.manifold import TSNE
                for i in range(similar_matrix.shape[-1]):
                    point_matrix[:, i] = TSNE(n_components=self.n_components, metric="precomputed").fit_transform(
                        similar_matrix[:, :, i]).reshape(point_matrix.shape[0])  # 每个相似度矩阵上降维
                # Change_Trend_Similarity = TSNE(n_components=1, metric="precomputed").fit_transform(similar_matrix[:, :, 0])
                # Position_Similarity = TSNE(n_components=1, metric="precomputed").fit_transform(similar_matrix[:, :, 1])
                # Timestamp_Similarity = TSNE(n_components=1, metric="precomputed").fit_transform(similar_matrix[:, :, 2])
                # point_matrix = np.concatenate([Change_Trend_Similarity, Position_Similarity, Timestamp_Similarity], axis=1)
            elif self.algorithm == 'ISOMAP':
                from sklearn.manifold import Isomap
                for i in range(similar_matrix.shape[-1]):
                    point_matrix[:, i] = Isomap(n_components=self.n_components, n_neighbors=3,
                                                metric="precomputed").fit_transform(
                                                similar_matrix[:, :, i]).reshape(point_matrix.shape[0])
            elif self.algorithm == 'MDS':
                from sklearn.manifold import MDS
                for i in range(similar_matrix.shape[-1]):
                    # point_matrix[:, i] = MDS(self.n_components).fit(
                    #   similar_matrix[:, :, i]).reshape(point_matrix.shape[0])
                    point_matrix[:, i] = MDS(n_components=self.n_components, dissimilarity="precomputed").fit_transform(
                        similar_matrix[:, :, i]).reshape(point_matrix.shape[0])
            elif self.algorithm == 'umap':
                import umap
                umap = umap.UMAP(n_components=self.n_components, metric="precomputed")
                for i in range(similar_matrix.shape[-1]):
                    point_matrix[:, i] = umap.fit_transform(similar_matrix[:, :, i]).reshape(point_matrix.shape[0])
            elif self.algorithm == 'SE':
                from sklearn.manifold import SpectralEmbedding
                SpectralEmbedding(n_components=self.n_components, affinity="precomputed").fit_transform()
            elif self.algorithm == 'LLE':
                from sklearn.manifold import LocallyLinearEmbedding
                LocallyLinearEmbedding(n_components=self.n_components).fit_transform()
            else:
                print("unknown algorithm!")
            # 数据归一化（方便后面聚类）
            for i in range(point_matrix.shape[-1]):
                point_matrix[:, i] = (point_matrix[:, i] - min(point_matrix[:, i])) / (
                        max(point_matrix[:, i]) - min(point_matrix[:, i]))
            print("Point_matrix normalization.")
            np.save(self.file_path, point_matrix)
            print('{} file saved.'.format(self.file_path))
        else:  # 不用重新生成数据, 直接读取文件，不需要重新计算全距离矩阵，且距离映射，因为解有很多个结果可能会有不同：
            point_matrix = np.load(self.file_path)
            print(" {} file read.".format(self.file_path))
        return point_matrix


if __name__ == '__main__':
    file = os.path.join(r'D:\Trajectory_analysis\Data', 'Similarity_matrix.npy')
    similarity_matrix = np.load(file)  # 上邻接矩阵，[轨迹变化相似度， 轨迹出发结束位置点，时间戳距离]
    sparse_matrix = np.full(similarity_matrix.shape, np.nan)  # 用来存放稀疏矩阵
    for i in range(similarity_matrix.shape[-1]):
        similarity_matrix[:, :, i] += similarity_matrix[:, :, i].T
        similarity_matrix[:, :, i] = (similarity_matrix[:, :, i] - min(similarity_matrix[:, :, i].flat)) / (max(
            similarity_matrix[:, :, i].flat) - min(similarity_matrix[:, :, i].flat))
    # 对数据进行筛选只保留d,t比较近的相似度矩阵
    K = 5
    # 对数据进行归一化
    for i in range(similarity_matrix.shape[0]):
        # 找到每个轨迹的最小K个（d，t）距离
        distance = np.zeros(similarity_matrix.shape[0])  # 存储轨迹与其他轨迹之间的距离
        for j in range(similarity_matrix.shape[0]):
            distance[j] = similarity_matrix[i, j, 1] ** 2 + similarity_matrix[i, j, 2] ** 2
        sort_index = np.argsort(distance)  # 对diatance进行排序
        for j in sort_index[1:1+K]:
            sparse_matrix[i, j, :] = similarity_matrix[i, j, :]
    # print(sparse_matrix)
    from sklearn.manifold import SpectralEmbedding
    SpectralEmbedding(n_components=1, affinity="precomputed_nearest_neighbors").fit_transform(sparse_matrix[:, :, 0])
    # np.save(r'D:\Trajectory_analysis\Output\point_matrix.npy', point_matrix)
