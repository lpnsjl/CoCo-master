"""
DNN神经网络
"""
import numpy as np
from Activator import *
from DNN.Layer import Layer
import matplotlib.pyplot as plt

class Model:
    def __init__(self, layers):
        """
        神经网络初始化
        :param layers:
        """
        self.layers = []  # 神经网络的全连接层
        # 构建神经网络
        for l in layers:
            layer = Layer(input_size=l.get("input_size"), output_size=l.get("output_size"),
                          forward_activator=l.get("forward_activator"), backward_activator=l.get("backward_activator"))
            self.layers.append(layer)
        self.history = []  # 存储神经网络训练历史数据

    def forward_propagation(self, X):
        """
        神经网络的前向传播
        :param X: 训练数据
        :return: 最后一层的输出
        """
        A = X
        for layer in self.layers:
           A = layer.forward_propagation(A)
        self.A = A  # 最后一层的输出, 也是神经网络的输出
        return A

    def backward_propagation(self, label):
        """
        神经网络的反向传播
        :param label: 训练标签
        :return:
        """
        dZ = self.A - label
        for layer in self.layers[::-1]:
            dZ = layer.backward_propagation(dZ)

    def update(self, alpha):
        """
        神经网络的权重更新
        :param alpha: 学习率
        :return:
        """
        for layer in self.layers:
            layer.update(alpha)

    def calc_loss(self, X, label):
        """
        计算损失函数
        :param X:
        :param label:
        :return: 损失函数的值
        """
        n, m = X.shape
        A = self.forward_propagation(X)
        L = -1*label.T@np.log(A)
        loss = np.sum(np.diag(L))/m
        return loss

    def train_one_batch(self, X, label, alpha):
        """
        小批量训练
        :param X: 小批量训练样本
        :param label: 小批量训练标签
        :param alpha: 学习率
        :return:
        """
        self.forward_propagation(X)
        self.backward_propagation(label)
        self.update(alpha)

    def train(self, X, label, batch_size, epochs, alpha):
        """
        神经网络训练
        :param X: 训练样本
        :param label: 训练标签
        :param batch_size: 批大小
        :param epochs: 迭代次数
        :param alpha: 学习率
        :return:
        """
        n, m = X.shape
        batchs = m//batch_size  # 训练样本可以分成多少批
        remain = m - batchs*batch_size
        for epoch in range(epochs):
            # 每迭代五次, 打印一次结果
            if epoch%5 == 0:
                #　loss = self.calc_loss(X, label)  # 交叉熵损失
                accuracy = self.evaluate(X, label)  # 预测准确率
                self.history.append([accuracy])  # 保存历史训练的精度与误差
                print("epoch{}-------------acurracy: {:.2%}".format(epoch, accuracy))
            for batch in range(batchs):
                batch_X = X[:, batch*batch_size: (batch+1)*batch_size]
                batch_label = label[:, batch*batch_size: (batch+1)*batch_size]
                self.train_one_batch(batch_X, batch_label, alpha)
            if remain > 0:
                batch_X = X[:, batchs * batch_size:]
                batch_label = label[:, batchs * batch_size:]
                self.train_one_batch(batch_X, batch_label, alpha)

    def predict(self, X):
        """
        神经网络预测
        :param X: 预测样本
        :return: 预测结果
        """
        A = self.forward_propagation(X)
        predict_label = np.argmax(A, axis=0)
        return predict_label

    def evaluate(self, X, label):
        """
        评估精度
        :param X:
        :param label:
        :return:
        """
        label = np.argmax(label, axis=0)
        predict_label = self.predict(X)
        size = len(label)
        correct_num = 0
        for i in range(size):
            if predict_label[i] == label[i]:
                correct_num += 1
        accuracy = correct_num/size
        return accuracy


# 数据转换
def convert(data, label):
    m = data.shape[0]
    data = data.reshape(m, -1).T
    class_num = len(set(label))
    labels = np.zeros((class_num, m))
    for i in range(m):
        labels[label[i], i] = 1
    return data, labels


# 加载训练数据与测试数据
def load_data():
    import tensorflow as tf
    mnist = tf.keras.datasets.mnist
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    train_data, train_label = convert(train_data, train_label)
    test_data, test_label = convert(test_data, test_label)
    return (train_data, train_label), (test_data, test_label)

if __name__ == "__main__":
    (train_data, train_label), (test_data, test_label) = load_data()
    # 初始化神经网络模型
    layers = [{"input_size": 784, "output_size":15, "backward_activator": ReLu, "forward_activator": ReLu},
              {"input_size": 15, "output_size": 10, "backward_activator": ReLu, "forward_activator": Softmax}]
    model = Model(layers)
    model.train(train_data[:, :60000], train_label[:, :60000], 64, 1000, 0.005)
    # 画图
    history = model.history
    size = len(history)
    accuracies = [h[0] for h in history]
    plt.plot(accuracies)
    plt.show()