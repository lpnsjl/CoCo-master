import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle


def euclideanDistance(x1, x2):
    """
    计算两个点之间的欧式距离
    :param x1: 点1的特征向量
    :param x2: 点2的特征向量
    :return: 两点之间的欧式距离
    """
    distance2 = np.sum((x1-x2)**2)
    return np.sqrt(distance2)


def randCent(X, k):
    """
    从样本集中随机取k个点作为质心
    :param X: 聚类样本
    :param k:
    :return:
    """
    new_X = np.copy(X)  # 深拷贝
    np.random.shuffle(new_X)
    return new_X[:k]


def kMeans(X, k):
    """
    kMean聚类方法
    :param k: 聚类数目
    :param X: 聚类样本
    :return: 聚类后的标签
    """
    m, n = X.shape
    # 从样本中随机选取k个中心点
    centroids = randCent(X, k)
    clusterAssign = np.zeros((m, 2))  # 存储聚类结果
    clusterChange = True  # 聚类改变标志
    while clusterChange:
        clusterChange = False
        for i in range(m):
            minDist = float('inf')
            minIndex = 0
            for j in range(k):
                dist = euclideanDistance(centroids[j], X[i])
                if dist < minDist:
                    minDist = dist
                    minIndex = j
            if clusterAssign[i, 0] != minIndex:
                clusterChange = True
            clusterAssign[i, :] = minIndex, minDist
        # 重新计算质心
        for cent in range(k):
            indices = (clusterAssign[:, 0]==cent)
            centroids[cent, :] = X[indices].mean(axis=0)
    # 打印聚类后的质心
    for cent in range(k):
        print("centroid_{}: {}".format(cent, centroids[cent]))
    # 计算kmeans算法的误差
    loss = clusterAssign[:, 1].mean()
    print("loss: {}".format(loss))
    # 聚类后的标签
    labels = clusterAssign[:, 0]
    return labels, loss, k


# 展示聚类结果
def show_cluster_result(X, labels, k):
    for i, col in zip(range(k), cycle('bgrcmyk')):
        indices = (labels == i)
        members = X[indices]
        x = members[:, 0]
        y = members[:, 1]
        plt.plot(x, y, col+'o')
    plt.show()


if __name__ == "__main__":
    from sklearn.datasets.samples_generator import make_blobs
    from sklearn.preprocessing import StandardScaler

    centers = [[1, 1, 1], [-1, -1, -1], [1, -1, 2]]
    X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)

    X = StandardScaler().fit_transform(X)
    labels, loss, k = kMeans(X, 3)
    show_cluster_result(X, labels, k)



