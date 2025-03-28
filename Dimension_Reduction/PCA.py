import numpy as np


class PCA:
    '''
    求主成分分析;
    threshold可选参数表示方差累计达到threshold后就不再取后面的特征向量.
    '''
    def __init__(self, dataset):
        # 每一列代表一个样本
        self.dataset = np.matrix(dataset, dtype='float64').T
        self.characteristic_value = []

    def principal_comps(self, threshold=0.85):
        ret = []
        data = []
        # 按行数据标准化
        for i in range(self.dataset.shape[0]):
            self.dataset[i] = (self.dataset[i] - np.mean(self.dataset[i]))/np.std(self.dataset[i], ddof=1)
        # 求协方差矩阵
        Cov = np.cov(self.dataset)
        # 求特征值和特征向量
        eigs, vectors = np.linalg.eig(Cov)
        # 第i个特征向量是第i列，为了便于观察将其转置一下
        for i in range(len(eigs)):
            data.append((eigs[i], vectors[:, i].T))
        # 按照特征值从大到小排序
        data.sort(key=lambda x: x[0], reverse=True)
        sum = 0
        for comp in data:
            sum += comp[0] / np.sum(eigs)
            ret.append(
                tuple(map(
                    lambda x: np.round(x, 5),
                    # 特征向量、方差贡献率、累计方差贡献率
                    (comp[1], comp[0] / np.sum(eigs), sum)
                ))
            )
            print('特征值:', comp[0], '特征向量:', ret[-1][0], '方差贡献率:', ret[-1][1], '累计方差贡献率:', ret[-1][2])
            self.characteristic_value.append(ret[-1][1])
            if sum > threshold:
                break
        print(ret)
        z = []
        for eigenvector, variance, Cumulative_variance in ret:
            z.append(eigenvector * self.dataset)
        z = np.array(z).reshape(len(ret), self.dataset.shape[1]).T
        return z

    def sklearn_pca(self, n=5):
        for i in range(self.dataset.shape[0]):
            self.dataset[i] = (self.dataset[i] - np.mean(self.dataset[i])) / np.std(self.dataset[i], ddof=1)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n).fit_transform(self.dataset.T)
        pca = PCA(n_components='mle')  # 第指定mle算法自动选取维数（mle极大似然估计）
        return pca


if __name__ == '__main__':
    p = PCA(
        [[66, 64, 65, 65, 65],
         [65, 63, 63, 65, 64],
         [57, 58, 63, 59, 66],
         [67, 69, 65, 68, 64],
         [61, 61, 62, 62, 63],
         [64, 65, 63, 63, 63],
         [64, 63, 63, 63, 64],
         [63, 63, 63, 63, 63],
         [65, 64, 65, 66, 64],
         [67, 69, 69, 68, 67],
         [62, 63, 65, 64, 64],
         [68, 67, 65, 67, 65],
         [65, 65, 66, 65, 64],
         [62, 63, 64, 62, 66],
         [64, 66, 66, 65, 67]]
    )

    lst = p.principal_comps()

    print(lst)
    a = p.sklearn_pca(2)
    print(a)