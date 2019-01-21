"""
cart决策树
"""
import numpy as np


class Node:
    def __init__(self, leftChild=None, rightChild=None, col=-1, value=None, data=None, isLeaf=False):
        """
        树节点
        :param leftChild: 左孩子
        :param rightChild: 右孩子
        :param col: 特征列
        :param value: 特征值
        :param data: 节点上的数据
        :param isLeaf: 是否是叶子节点
        """
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.col = col
        self.value = value
        self.data = data
        self.isLeaf = isLeaf


class Model:
    def __init__(self, train_X, train_y, mini_gini):
        """
        cart模型初始化
        :param train_X: 训练样本
        :param train_y: 训练标签
        :param mini_gini: gini指数阀值, 用于剪枝
        """
        self.train_X = train_X
        self.train_y = train_y
        self.mini_gini = mini_gini




