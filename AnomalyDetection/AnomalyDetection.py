import numpy as np
from scipy import io

class AnomalyDetection(object):
    def __init__(self, x):
        self.x = x


    def etismateParam(self):
        """
        参数估计, 计算样本均值与方差,单元高斯分布
        :return: 均值mu, 方差sigma
        """
        mu = np.mean(self.x, axis=0)
        sigma = np.var(self.x, axis=0)
        return mu, sigma

    def selectThreshold(self, pval, yval):
        """
        选取最好的threshold
        :param pval: 交叉验证集中预测样本的值
        :param yval: 交叉验证集中样本的真实值
        :return: 最佳的threshld与最佳的F1Score
        """
        # 计算threshold选取步长
        bestThreshold = 0
        bestF1Score = 0
        step = (np.max(pval) - np.min(pval))/1000
        for threshold in np.arange(np.min(pval), np.max(pval), step):
            cvProbability = (pval<threshold) # 获取正样本与负样本
            tp = np.sum((cvProbability == 1)&(yval == 1).ravel()).astype(float) # 真实值为正样本且预测正确样本个数
            tf = np.sum((cvProbability == 1)&(yval == 0).ravel()).astype(float) # 真实值为负样本却预测错误样本个数
            tn = np.sum((cvProbability == 0)&(yval == 1).ravel()).astype(float) # 真实值为正样本但预测错误样本个数
            precision = tp/(tp+tf) # 精准度
            recall = tp/(tp+tn) # 召回率
            F1Score = 2*precision*recall/(precision+recall) # F1Score检验不同threshold的异常点检测的效果
            print(threshold, tp, tf, tn, precision, F1Score)
            if(F1Score>bestF1Score):
                bestF1Score = F1Score
                bestThreshold = threshold
        return bestThreshold, bestF1Score

    def oneGaussian(self, cx):
        """
        单元高斯分布密度计算函数
        :param cx: 交叉验证集样本
        :return: 交叉验证集样本对应的高斯密度概率pval
        """
        mu, sigma = self.etismateParam()
        m, n = cx.shape
        pval = np.zeros(m)
        for i in range(m):
            pval[i] = innermul(np.exp(-(cx[i] - mu)**2/2*sigma)/(2*np.pi*np.sqrt(sigma)))
        return pval

def innermul(a):
    """
    一维数组所有内部元素乘积
    :param a: 一维数组
    :return: 数组乘积
    """
    num = 1
    for i in a:
        num *= i
    return num

if __name__ == "__main__":
    data = io.loadmat("/home/sjl/桌面/machine learning/mlclass-ex8-005/mlclass-ex8/ex8data1.mat")
    x = data['X']
    yval = data['yval']
    xval = data['Xval']
    anomal = AnomalyDetection(x)
    pval = anomal.oneGaussian(xval)
    bestThreshold, bestF1Score = anomal.selectThreshold(pval, yval)
    print(bestF1Score, bestThreshold)
