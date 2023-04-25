#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File: perceptron.py
@Time: 2023/2/16-12:45
@Author: Li Dongchao
@Desc: 感知机模型
"""

import numpy as np


class Perceptron(object):
    """感知机模型"""

    def __init__(self, train_x, train_y, learning_rate: float = 1.0):
        if 1 < learning_rate <= 0:
            raise ValueError("学习率的取值应该位于区间(0, 1]内！")
        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)
        self.w = None
        self.b = None
        self.init_params()
        self.learning_rate = learning_rate
        self.gram_matrix = None
        self.a = np.zeros((self.sample_nums, 1))

    @property
    def feature_nums(self):
        """训练集特征数量"""
        return self.train_x.shape[1]

    @property
    def sample_nums(self):
        """样本数量"""
        return self.train_x.shape[0]

    def init_params(self, init_w=None, init_b=None):
        """初始化模型参数"""
        if init_w is None:
            self.w = np.zeros((self.feature_nums, 1))
        if init_b is None:
            self.b = 0

    def is_classify_wrong(self, x, y):
        """判断样本的分类是否错误"""
        return (y * (np.dot(x, self.w).flatten()[0] + self.b)) <= 0

    def gradient_descent(self):
        """梯度下降算法"""
        i = 0
        while i < self.sample_nums:
            cur_x = self.train_x[i, :].reshape(1, -1)
            cur_y = self.train_y[i]
            if self.is_classify_wrong(cur_x, cur_y):
                delta_w = self.delta_w(cur_x, cur_y)
                delta_b = self.delta_b(cur_y)
                self.w = self.w + self.learning_rate * delta_w
                self.b = self.b + self.learning_rate * delta_b
                i = 0
            else:
                i += 1

    def cal_gram_matrix(self):
        """计算gram矩阵"""
        self.gram_matrix = np.dot(self.train_x, np.transpose(self.train_x))
        return self.gram_matrix

    def is_classify_wrong_duality(self, i):
        """计算对偶形式中样本是否分类错误"""
        return (self.train_y[i] * (np.dot(self.gram_matrix[i, :], self.a * self.train_y.reshape(-1, 1))[0] + self.b)) <= 0

    def gradient_descent_duality(self):
        """对偶问题的梯度下降"""
        self.cal_gram_matrix()
        i = 0
        while i < self.sample_nums:
            if self.is_classify_wrong_duality(i):
                self.a[i] = self.a[i] + self.learning_rate
                self.b = self.b + self.learning_rate * self.train_y[i]
                i = 0
            else:
                i += 1

    @staticmethod
    def delta_w(x, y):
        """计算w的导数"""
        return np.transpose(x) * y

    @staticmethod
    def delta_b(y):
        """计算b的导数"""
        return y


if __name__ == '__main__':
    data = [(3, 3, 1), (4, 3, 1), (1, 1, -1)]
    data_x = [x[0: 2] for x in data]
    data_y = [x[-1] for x in data]
    model = Perceptron(data_x, data_y)
    model.gradient_descent_duality()
    print(model.a, model.b)
