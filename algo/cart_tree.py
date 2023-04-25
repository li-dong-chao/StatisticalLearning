#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File: cart_tree.py
@Time: 2023/3/22-12:49
@Author: Li Dongchao
@Desc: CART树
"""

from queue import Queue
import numpy as np
from typing import Any, Self, Union, List, Tuple
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.colors as colors

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
color_list = list(colors.BASE_COLORS.keys())


class Node(object):
    """cart树的节点，根节点即为cart树"""

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        self.left: Union[Self, None] = None  # 左子树
        self.right: Union[Self, None] = None  # 右子树
        self.split_feature: Union[int, None] = None  # 该节点的最有特征
        self.split_value: Union[Any, None] = None  # 该节点的最有划分值
        self.parent: Union[Self, None] = None  # 该节点的父节点
        self.predict_value: Union[Any, None] = None  # 该节点的预测结果
        self.train_y = train_y
        self.train_x = train_x
        self.available_features: List[int] = list(range(train_x.shape[1]))  # 候选特征集

    @property
    def is_leaf(self) -> bool:
        """判断一个树是否为叶节点"""
        return self.left is None and self.right is None

    def set_predict_value(self):
        """获取当前节点的预测值"""
        if self.is_continuous(self.train_y):
            self.predict_value = self.train_y.mean()
        else:
            self.predict_value = Counter(self.train_y.flatten()).most_common(1)[0][0]

    @classmethod
    def cal_gain(cls, y: np.ndarray) -> float:
        """计算基尼指数"""
        counter = np.array(list(Counter(y.flatten()).values()))  # 统计各个类别的数量
        probs = counter / counter.sum()  # 计算各个类别的概率
        gini = 1 - (probs ** 2).sum()  # 计算gini指数
        return gini

    def gini(self) -> float:
        """计算当前节点基尼指数 :math:`Gini(D)` """
        return self.cal_gain(self.train_y)

    def cal_gini_split(self, feature: int, value: Any) -> float:
        """计算分割后的基尼指数 :math:`Gini(D, A)` """
        left_data = self.train_y[self.train_x[:, feature] == value, :]  # D1
        right_data = self.train_y[self.train_x[:, feature] != value, :]  # D2
        gini1 = self.cal_gain(left_data)  # Gini(D1)
        gini2 = self.cal_gain(right_data)  # Gini(D2)
        gini_split = (left_data.shape[0] * gini1 +
                      right_data.shape[0] * gini2) / self.train_y.shape[0]  # Gini(D, A)
        return gini_split

    @classmethod
    def cal_square_error(cls, y: np.ndarray) -> float:
        """计算平方误差"""
        r = y.var() * y.shape[0]
        return r

    def square_error(self):
        """计算当前节点的平方误差"""
        return self.cal_square_error(self.train_y)

    def cal_square_error_split(self, feature: int, value: float) -> float:
        """计算给定分割下的损失"""
        left_data = self.train_y[self.train_x[:, feature] <= value, :]
        right_data = self.train_y[self.train_x[:, feature] > value, :]
        square_error1 = self.cal_square_error(left_data)
        square_error2 = self.cal_square_error(right_data)
        square_error_split = square_error1 +square_error2
        return square_error_split

    @classmethod
    def is_continuous(cls, x: np.ndarray):
        """判断一个变量是不是连续变量"""
        return x.dtype.kind in ["l", "f"]  # warning: 这里判断可能不充分

    def find_best_split(self) -> Tuple[int, Any]:
        """寻找最有分割特征和最优分割值"""
        best_feature = None
        best_value = None
        min_loss = np.inf
        for feature in self.available_features:
            if self.is_continuous(self.train_x[:, feature]):
                for value in np.unique(self.train_x[:, feature]):
                    cur_loss = self.cal_square_error_split(feature, value)
                    if cur_loss < min_loss:
                        best_feature = feature
                        best_value = value
            else:
                for value in np.unique(self.train_x[:, feature]):
                    cur_loss = self.cal_gini_split(feature, value)
                    if cur_loss < min_loss:
                        min_loss = cur_loss
                        best_feature = feature
                        best_value = value
        return best_feature, best_value

    def fit(self, min_sample: int, min_gini: float) -> None:
        """
        训练cart树
        :param min_sample: 最小样本量
        :param min_gini: 最小基尼系数
        :return:
        """
        # 样本数或基尼系数小于给定参数阈值，结束
        if self.train_y.shape[0] < min_sample or self.gini() < min_gini:
            self.set_predict_value()
            return
        # (1) D中的所有实例都属于同一类 C_k，返回
        if len(np.unique(self.train_y)) == 1:
            self.set_predict_value()
            return
        # (2) A为空集，返回
        if len(self.available_features) == 0:
            self.set_predict_value()
            return
        best_feature, best_value = self.find_best_split()
        if best_feature is None:
            self.set_predict_value()
            return
        self.split_feature = best_feature
        self.split_value = best_value
        left_x = self.train_x[self.train_x[:, best_feature] == best_value, :]
        left_y = self.train_y[self.train_x[:, best_feature] == best_value]
        right_x = self.train_x[self.train_x[:, best_feature] != best_value, :]
        right_y = self.train_y[self.train_x[:, best_feature] != best_value]
        left_node = Node(left_x, left_y)
        right_node = Node(right_x, right_y)
        self.available_features.remove(best_feature)
        left_node.available_features = self.available_features
        left_node.parent = self
        right_node.available_features = self.available_features
        right_node.parent = self
        self.left = left_node
        self.right = right_node
        self.set_predict_value()
        left_node.fit(min_sample, min_gini)
        right_node.fit(min_sample, min_gini)

    def width_traversal(self):
        """决策树的宽度优先遍历"""
        q = Queue()
        q.put(self)
        height = 0
        end = self
        info = []
        while not q.empty():
            head: Self = q.get()
            info.append((height, head))
            if head.left is not None:
                q.put(head.left)
            if head.right is not None:
                q.put(head.right)
            if end == head:
                height += 1
                if not q.empty():
                    end = q.queue[-1]
        return info

    def show(self, feature_name: list = None):
        """绘制决策树"""
        if feature_name is None:  # noqa
            feature_name = list(range(self.train_x.shape[1]))
        info = self.width_traversal()
        width_info = [x for x, _ in info]
        node_info = [x for _, x in info]
        counter = Counter(width_info)
        interval = 20
        height = len(counter) * interval
        width = counter.most_common(1)[0][1] * interval
        points = []
        for k, v in counter.items():
            h = height - k * interval
            cur_interval = width / (v + 1)
            cur_points = [(cur_interval + i * cur_interval, h) for i in range(v)]
            cur_points.sort(key=lambda x: x[0])
            points.extend(cur_points)
        points.sort(key=lambda x: x[1], reverse=True)
        mapper = {}
        categories = []
        features = []
        values = []
        for i, node in enumerate(node_info):
            mapper.update({node: points[i]})
            values.append(node.split_value)
            categories.append(node.predict_value)
            cur_feature = feature_name[node.split_feature] if node.split_feature is not None else ""
            features.append(cur_feature)
        color = {x: i + 1 for i, x in enumerate(set(categories))}
        lines = []
        line_text = []
        for k, v in mapper.items():
            if k.left is not None:
                lines.append((v, mapper.get(k.left)))
                line_text.append("等于")
            if k.right is not None:
                lines.append((v, mapper.get(k.right)))
                line_text.append("不等于")
        for i, (x, y) in enumerate(points):
            plt.scatter(x, y, c=color_list[color[categories[i]]])
            plt.text(x - 1, y - 2, f"value: {values[i]}")
            plt.text(x - 1, y - 4, f"predict: {categories[i]}")
            plt.text(x - 1, y - 6, f"feature: {features[i]}")
        for i, (x, y) in enumerate(lines):  # noqa
            dx = y[0] - x[0]
            dy = y[1] - x[1]
            plt.arrow(x[0], x[1], dx, dy)
            plt.text(x[0] + dx / 2, x[1] + dy / 2, line_text[i])
        plt.show()


if __name__ == '__main__':
    datasets = [['青年', '否', '否', '一般', '否'],
                ['青年', '否', '否', '好', '否'],
                ['青年', '是', '否', '好', '是'],
                ['青年', '是', '是', '一般', '是'],
                ['青年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '好', '否'],
                ['中年', '是', '是', '好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '好', '是'],
                ['老年', '是', '否', '好', '是'],
                ['老年', '是', '否', '非常好', '是'],
                ['老年', '否', '否', '一般', '否']]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    df = np.array(datasets)
    train_X = df[:, :-1]
    train_Y = df[:, -1].reshape((-1, 1))
    tree = Node(train_X, train_Y)
    tree.fit(min_sample=4, min_gini=0)
    tree.show(feature_name=labels)
