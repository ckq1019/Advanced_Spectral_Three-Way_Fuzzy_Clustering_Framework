import numpy as np
import os


def Consensus_similarity(similarity_matrices):
    '''
    根据论文Multi-view spectral clustering via sparse graph learning 得到一致性相似度
    min sum(view_similarity - consensus_similarity) + lambda * linalg(consensus_similarity, 0)
    :param similarity_matrices:
    :return:consensus_similarity
    '''
    W = np.zeros((similarity_matrices.shape[0], similarity_matrices.shape[1]))
    k = similarity_matrices.shape[-1]  # 视图个数
    lam = 1  # 视图权重
    for v in range(k):
        W += similarity_matrices[:, :, v]

    def S(mu, x):  # 逐元法
        for row in range(x.shape[0]):
            for col in range(x.shape[1]):
                if x[row][col] > mu:
                    x[row][col] = x[row][col] - mu
                elif x[row][col] < -mu:
                    x[row][col] = x[row][col] + mu
                else:
                    x[row][col] = 0
        return x

    return S(lam / (2 * k), W / k)


if __name__ == '__main__':
    file_path = r"cyclonic_[353  80].csv"
    from main import read_data

    origin_data, file_format = read_data(os.path.join(r'D:\Trajectory_analysis\Data', file_path))
    from Data_Preprocessing import format_conversion
    data = format_conversion(origin_data, file_format='CSV', Data_processing=False)

    track_list = []
    for i in data.keys():
        track_list.append(data[i])
    from Similarity.Similarity_Measurement_Methodology import SimilarityMatrix
    sm = SimilarityMatrix(isCalculate=False, TR_List=track_list, K=0, reshuffle=False,
                          filepath=os.path.join(r'D:\Trajectory_analysis\Similarity\output',
                                                'All_Similarity_matrix.npy'))
    similar_matrix = sm.similarity_matrix
    W = np.full(len(track_list), 1)
    consensus_similarity = Consensus_similarity(similar_matrix)
    from Cluster.SC import mySpectral_Clustering
    mySpectral_Clustering(10, consensus_similarity, W)  # 返回各个簇的隶属度
