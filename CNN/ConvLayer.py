"""
卷积层实现
"""
import numpy as np
from Activator import *

# 卷积操作, 自动区分2D与3D
def conv(input_array, kernel_array, stride=1, bias=0):
    if input_array.ndim == 2:
        input_height, input_width = input_array.shape
        kernel_height, kernel_width = kernel_array.shape
        output_height = (input_height-kernel_height)//stride + 1
        output_width = (input_width-kernel_width)//stride + 1
        output_array = np.zeros((output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                conv_patch = input_array[i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width]
                output_array[i, j] = (conv_patch*kernel_array).sum() + bias
        return output_array

    elif input_array.ndim == 3:
        input_depth, input_height, input_width = input_array.shape
        kernel_depth, kernel_height, kernel_width = kernel_array.shape
        output_height = (input_height - kernel_height) // stride + 1
        output_width = (input_width - kernel_width) // stride + 1
        output_array = np.zeros((output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                conv_patch = input_array[:, i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width]
                output_array[i, j] = (conv_patch*kernel_array).sum() + bias
        return output_array

# 数组填充
def pad(input_array, zero_padding):
    if input_array.ndim == 2:
        input_height, input_width = input_array.shape
        padded_input_array = np.zeros((input_height + 2*zero_padding, input_width + 2*zero_padding))
        padded_input_array[zero_padding: zero_padding+input_height,
        zero_padding: zero_padding+input_width] = input_array
        return padded_input_array
    elif input_array.ndim == 3:
        input_depth, input_height, input_width = input_array.shape
        padded_input_array = np.zeros((input_depth, input_height + 2 * zero_padding, input_width + 2 * zero_padding))
        padded_input_array[:, zero_padding: zero_padding + input_height,
        zero_padding: zero_padding + input_width] = input_array
        return padded_input_array


# 卷积核
class Kernel:
    def __init__(self, depth, kernel_height, kernel_width):
        """
        卷积核初始化
        :param depth: 卷积核信道数
        :param kernel_height: 卷积核高
        :param kernel_width: 卷积核宽
        """
        self.W = np.random.uniform(-1e-4, 1e-4, (depth, kernel_height, kernel_width))
        self.b = 0
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = 0

    def update(self, alpha):
        """
        卷积核更新
        :param alpha: 学习率
        :return:
        """
        self.W -= alpha*self.grad_W
        self.b -= alpha*self.grad_b


class ConvLayer:
    def __init__(self, input_height, input_width, channel, kernel_height, kernel_width, kernel_num, backward_activator,
                 forward_activator, stride=1, zero_padding=0):
        """
        卷积层初始化
        :param input_height: 输入高度
        :param input_width: 输入宽度
        :param channel: 信道数
        :param kernel_height: 卷积核高
        :param kernel_width: 卷积核宽
        :param kernel_num: 卷积核数目
        :param backward_activator: 反向传播激活函数
        :param forward_activator: 前向传播激活函数
        :param stride: 步幅
        :param zero_padding: 零填充大小
        """
        self.input_height = input_height
        self.input_width = input_width
        self.channel = channel
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.kernel_num = kernel_num
        self.stride = stride
        self.zero_padding = zero_padding
        self.backward_activator = backward_activator()
        self.forward_activator = forward_activator()
        self.kernels = []  # 存储卷积核
        for k in range(kernel_num):
            self.kernels.append(Kernel(channel, kernel_height, kernel_width))

    def forward_propagation(self, input_array):
        """
        卷积层前向传播
        :param input_array: 输入数组
        :return:
        """
        self.input_array = input_array
        padded_input_array = pad(input_array, self.zero_padding)  # 零填充数组
        output_height = (self.input_height+2*self.zero_padding-self.kernel_height)//self.stride + 1
        output_width = (self.input_width+2*self.zero_padding-self.kernel_width)//self.stride + 1
        output_array = np.zeros((self.kernel_num, output_height, output_width))
        for k in range(self.kernel_num):
            kernel = self.kernels[k]
            output_array[k] = self.forward_activator.forward_propagation(
                conv(padded_input_array, kernel.W, self.stride, kernel.b))
        return output_array

    def backward_propagation(self, sensitivity_map):
        """
        卷积层反向传播
        :param sensitivity_map: 后一层的误差
        :return: 前一层的误差
        """
        expanded_sensitivity_map = self.expand_sensitivity_map(sensitivity_map)
        expanded_height = expanded_sensitivity_map.shape[1]
        zp = (self.kernel_height+self.input_height-expanded_height-1)//2
        padded_sensitivity_map = pad(expanded_sensitivity_map, zp)
        padded_input_array = pad(self.input_array, self.zero_padding)
        delta_array = np.zeros((self.channel, self.input_height, self.input_width))  # 初始化前一层误差
        for k in range(self.kernel_num):
            kernel = self.kernels[k]
            for d in range(self.channel):
                delta_array[d] += conv(padded_sensitivity_map[k], np.rot90(kernel.W[d], 2), 1, 0)
                kernel.grad_W[d] = conv(padded_input_array[d], expanded_sensitivity_map[k], 1, 0)
            kernel.grad_b = expanded_sensitivity_map[k].sum()
        delta_array = self.backward_activator.backward_propagation(delta_array)
        return delta_array

    def expand_sensitivity_map(self, sensitivity_map):
        """
        扩展误差图
        :param sensitivity_map:
        :return:
        """
        depth = sensitivity_map.shape[0]
        expanded_sensitivity_map = np.zeros((depth, self.input_height+2*self.zero_padding-self.kernel_height+1,
                                             self.input_width+2*self.zero_padding-self.kernel_width+1))

        output_height = (self.input_height + 2 * self.zero_padding - self.kernel_height) // self.stride + 1
        output_width = (self.input_width + 2 * self.zero_padding - self.kernel_width) // self.stride + 1
        for i in range(output_height):
            for j in range(output_width):
                expanded_sensitivity_map[:, i*self.stride, j*self.stride] = sensitivity_map[:, i, j]
        return expanded_sensitivity_map

    def update(self, alpha):
        """
        卷积层更新
        :param alpha:
        :return:
        """
        for kernel in self.kernels:
            kernel.update(alpha)


if __name__ == "__main__":
    # 梯度检查, 结果显示梯度正确
    a = np.array(
        [[[0, 1, 1, 0, 2],
          [2, 2, 2, 2, 1],
          [1, 0, 0, 2, 0],
          [0, 1, 1, 0, 0],
          [1, 2, 0, 0, 2]],
         [[1, 0, 2, 2, 0],
          [0, 0, 0, 2, 0],
          [1, 2, 1, 2, 1],
          [1, 0, 0, 0, 0],
          [1, 2, 1, 1, 1]],
         [[2, 1, 2, 0, 0],
          [1, 0, 0, 1, 0],
          [0, 2, 1, 0, 1],
          [0, 1, 2, 2, 2],
          [2, 1, 0, 0, 1]]])
    b = np.array(
        [[[0, 1, 1],
          [2, 2, 2],
          [1, 0, 0]],
         [[1, 0, 2],
          [0, 0, 0],
          [1, 2, 1]]])
    cl = ConvLayer(5, 5, 3, 3, 3, 2, Identity, Identity, 2, 1)
    cl.kernels[0].W = np.array(
        [[[-1, 1, 0],
          [0, 1, 0],
          [0, 1, 1]],
         [[-1, -1, 0],
          [0, 0, 0],
          [0, -1, 0]],
         [[0, 0, -1],
          [0, 1, 0],
          [1, -1, -1]]], dtype=np.float64)
    cl.kernels[0].b = 1
    cl.kernels[1].W = np.array(
        [[[1, 1, -1],
          [-1, -1, 1],
          [0, -1, 1]],
         [[0, 1, 0],
          [-1, 0, -1],
          [-1, 1, 0]],
         [[-1, 0, 0],
          [-1, 0, 1],
          [-1, 0, 0]]], dtype=np.float64)

    error_function = lambda o: o.sum()
    output_array = cl.forward_propagation(a)
    print(output_array)
    sensitivity_array = np.ones(output_array.shape,
                                dtype=np.float64)
    delta_array = cl.backward_propagation(sensitivity_array)
    print(delta_array)

    epolison = 10e-4
    for d in range(cl.kernels[0].W.shape[0]):
        for i in range(cl.kernels[0].W.shape[1]):
            for j in range(cl.kernels[0].W.shape[2]):
                cl.kernels[0].W[d, i, j] += epolison
                output_array = cl.forward_propagation(a)
                err1 = error_function(output_array)
                cl.kernels[0].W[d, i, j] -= 2*epolison
                output_array = cl.forward_propagation(a)
                err2 = error_function(output_array)
                expected_grad = (err1 - err2)/ (2*epolison)
                cl.kernels[0].W[d, i, j] += epolison
                print('weights(%d,%d,%d): expected - actural %f - %f' % (
                    d, i, j, expected_grad, cl.kernels[0].grad_W[d,i,j]))