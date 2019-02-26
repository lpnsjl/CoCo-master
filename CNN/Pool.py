"""
池化层
"""
import numpy as np


# 获取一个2D区域的最大值所在的索引
def get_max_index(array):
    max_i = 0
    max_j = 0
    max_value = array[0,0]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i,j] > max_value:
                max_value = array[i,j]
                max_i, max_j = i, j
    return max_i, max_j


class MaxPool:
    def __init__(self, input_height, input_width, channel, filter_height, filter_width, stride):
        self.input_width = input_width
        self.input_height = input_height
        self.channel = channel
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride
        self.output_width = (input_width -
                             filter_width) // self.stride + 1
        self.output_height = (input_height -
                              filter_height) // self.stride + 1
        self.output_array = np.zeros((self.channel,
                                      self.output_height, self.output_width))

    def forward_propagation(self, input_array):
        self.input_array = input_array
        for d in range(self.channel):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output_array[d, i, j] = input_array[d, i*self.stride:i*self.stride+self.filter_height,
                                                 j*self.stride:j*self.stride+self.filter_width].max()
        return self.output_array

    def backward_propagation(self, sensitivity_array):
        delta_array = np.zeros(self.input_array.shape)
        for d in range(self.channel):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    k, l = get_max_index(self.input_array[d, i*self.stride:i*self.stride+self.filter_height,
                                                 j*self.stride:j*self.stride+self.filter_width])
                    delta_array[d, i * self.stride + k, j * self.stride + l] = sensitivity_array[d, i, j]
        return delta_array

    def update(self, alpha):
        pass


if __name__ == "__main__":
    # maxpool层检查
    a = np.array(
        [[[1, 1, 2, 4],
          [5, 6, 7, 8],
          [3, 2, 1, 0],
          [1, 2, 3, 4]],
         [[0, 1, 2, 3],
          [4, 5, 6, 7],
          [8, 9, 0, 1],
          [3, 4, 5, 6]]], dtype=np.float64)

    b = np.array(
        [[[1, 2],
          [2, 4]],
         [[3, 5],
          [8, 2]]], dtype=np.float64)

    mpl = MaxPool(4, 4, 2, 2, 2, 2)

    mpl.forward_propagation(a)
    print('input array:\n%s\noutput array:\n%s' % (a,
                                             mpl.output_array))
    delta_array = mpl.backward_propagation(b)
    print('input array:\n%s\nsensitivity array:\n%s\ndelta array:\n%s' % (
        a, b, delta_array))