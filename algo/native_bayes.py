#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File: native_bayes.py
@Time: 2023/2/26-14:09
@Author: Li Dongchao
@Desc: 
"""

import numpy as np
from collections import Counter, OrderedDict


class BayesClassifier(object):
    """朴素贝叶斯分类器"""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y
        self.num_sample, self.num_feature = x.shape
        self.class_list = np.unique(self.y)
        self.priori_prob = {}
        self.conditional_prob = {}
        self.alpha = 0

    def cal_y_prob(self):
        """计算Y的先验概率"""
        n = self.y.shape[0]
        static_result = Counter(self.y)
        for k, v in static_result.items():
            self.priori_prob.update({k: (v + self.alpha) * 1.0 / (n + len(self.class_list) * self.alpha)})

    def cal_x_prob(self):
        """计算X的条件概率"""
        for k in self.class_list:
            for j in range(self.num_feature):
                tmp_x = self.x[self.y == k, j]
                x_static_result = Counter(tmp_x)
                n = len(tmp_x)
                for m, s in x_static_result.items():
                    self.conditional_prob.update({
                        (k, j, m): (s + self.alpha) * 1.0 / (n + len(x_static_result) * self.alpha)
                    })

    def fit(self, alpha: float = 0):
        """训练模型"""
        self.alpha = alpha
        self.cal_y_prob()
        self.cal_x_prob()

    def k_prob(self, x: np.ndarray, k):
        """计算x在第k个类下的概率"""
        prob = self.priori_prob.get(k, 0)
        for j, value in enumerate(x.flatten()):
            prob *= self.conditional_prob.get((k, j, value), 0)
        return prob

    def predict(self, x: np.ndarray):
        """使用训练好的模型预测结果"""
        res = OrderedDict()
        for k in self.class_list:
            res.update({k: self.k_prob(x, k)})
        return list(res.keys())[0]


if __name__ == '__main__':
    train_data = np.array(
        [
            [1, "S", -1],
            [1, "M", -1],
            [1, "M", 1],
            [1, "S", 1],
            [1, "S", -1],
            [2, "S", -1],
            [2, "M", -1],
            [2, "M", 1],
            [2, "L", 1],
            [2, "L", 1],
            [3, "L", 1],
            [3, "M", 1],
            [3, "M", 1],
            [3, "L", 1],
            [3, "L", -1],
        ]
    )
    model = BayesClassifier(train_data[:, :2], train_data[:, -1])
    model.fit(alpha=1)
    print(model.predict(np.array([2, 'S'])))
