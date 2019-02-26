"""
flatten层
"""
import numpy as np


class Flatten:
    def forward_propagation(self, input_array):
        """
        faltten层前向传播
        :param input_array:
        :return:
        """
        self.input_array = input_array
        return input_array.flatten()

    def backward_propagation(self, sensitivity_array):
        """
        flatten层反向传播
        :param sensitivity_array:
        :return:
        """
        if self.input_array.ndim == 2:
            height, width = self.input_array.shape
            return sensitivity_array.reshape(height, width)
        elif self.input_array.ndim == 3:
            depth, height, width = self.input_array.shape
            return sensitivity_array.reshape(depth, height, width)

    def update(self, alpha):
        """
        flatten层更新, 实际上不需要更新
        :param alpha:
        :return:
        """
        pass