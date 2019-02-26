"""
实现各种激活函数
"""
import numpy as np


# sigmoid激活函数
class Sigmoid:
    @staticmethod
    def forward_propagation(Z):
        return 1/(1+np.exp(-1*Z))

    @staticmethod
    def backward_propagation(A):
        return A*(1-A)


# tanh激活函数
class Tanh:
    @staticmethod
    def forward_propagation(Z):
        return (np.exp(Z)-np.exp(-1*Z))/(np.exp(Z)+np.exp(-1*Z))

    @staticmethod
    def backward_propagation(A):
        return 1-A**2


# ReLu激活函数
class ReLu:
    @staticmethod
    def forward_propagation(Z):
        index = (Z<0)
        Z[index] = 0
        return Z

    @staticmethod
    def backward_propagation(A):
        index = (A>0)
        A[index] = 1
        return A

# IdentityActivator激活函数
class Identity:
    @staticmethod
    def forward_propagation(Z):
        return Z
    @staticmethod
    def backward_propagation(A):
        return np.ones(A.shape)

# softmax激活函数
class Softmax:
    @staticmethod
    def forward_propagation(Z):
        Z = np.exp(Z)
        return Z/np.sum(Z, axis=0)