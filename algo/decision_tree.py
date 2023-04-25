#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File: decision_tree.py
@Time: 2023/3/6-13:12
@Author: Li Dongchao
@Desc: 
"""

from enum import Enum
from queue import Queue
import numpy as np
from typing import Dict, Optional, Self, List
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.colors as colors

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
color_list = list(colors.BASE_COLORS.keys())


class TreeType(Enum):
    """决策树类型"""
    ID3 = "ID3"
    C45 = "C45"
    CART = "CART"


class DecisionTree(object):
    """决策树分类器"""

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
    ):
        # todo 增加assert，对数据的维度进行验证
        self.train_x: np.ndarray = x
        self.train_y: np.ndarray = y
        self.feature: Optional[int] = None  # 该节点特征（索引），即分类效果最好的特征，对应 A_g
        self.children: Dict[str, Self] = {}  # 该节点的子节点，保存在一个字典里面，字典的值为self.feature的取值
        self.available_features: List[int] = list(range(x.shape[1]))  # 剩余的特征，对应 A
        self.parent = None
        self.is_leaf = True  # 该节点是否为叶节点
        self.category = None  # 类别

    @property
    def sample_num(self):
        """当前节点样本总量"""
        return self.train_x.shape[0]

    def set_category(self):
        """获取当前节点的类别"""
        self.category = Counter(self.train_y.flatten()).most_common(1)[0][0]

    def get_classify_probs(self):
        """计算各个类别的概率"""
        counter = np.array(list(Counter(self.train_y.flatten()).values()))
        probs = counter / self.train_y.shape[0]
        return np.array(probs).reshape(1, -1)

    @classmethod
    def cal_entropy(cls, probs: np.ndarray) -> float:
        """根据概率值计算熵"""
        # 处理概率为0的情况，copy一个概率数组，然后修改0元素的值为1，求log时用这个修改后的值，不会影响结果
        probs = probs.reshape((1, -1))
        probs_copy = probs.copy()
        probs_copy[probs_copy == 0] = 1
        return -np.dot(probs, np.log2(np.transpose(probs_copy)))[0][0]

    def cal_info_gain(self, feature):
        """计算特征feature的信息增益"""
        classify_probs = self.get_classify_probs()
        entropy1 = self.cal_entropy(classify_probs)
        entropy2 = self.cal_condition_entropy(feature)
        return entropy1 - entropy2

    def cal_info_gain_rate(self, feature):
        """计算特征feature的信息增益比（信息增益率）"""
        gain = self.cal_info_gain(feature)
        p = self.get_feature_probs(feature)
        h = self.cal_entropy(p)
        return gain / h

    def get_feature_probs(self, feature):
        """计算训练集关于特征feature取值的熵，即 :math:`H_A(D)` """
        x = self.train_x[:, feature]
        counter = np.array(list(Counter(x.flatten()).values()))
        probs = counter / x.shape[0]
        return np.array(probs).reshape(1, -1)

    def cal_condition_entropy(self, feature):
        """计算经验条件熵"""
        h = 0
        counter1 = Counter(self.train_x[:, feature].flatten())
        for value, count in counter1.items():
            cur_y = self.train_y[self.train_x[:, feature] == value]
            counter = np.array(list(Counter(cur_y.flatten()).values()))
            probs = counter / cur_y.shape[0]
            h += count / self.train_y.shape[0] * self.cal_entropy(probs)
        return h

    def fit(self, epsilon: float, tree_type: TreeType = "ID3"):
        """构建决策树"""
        # (1) D中的所有实例都属于同一类 C_k，返回
        if len(np.unique(self.train_y)) == 1:
            self.set_category()
            return self
        # (2) A为空集，返回
        if len(self.available_features) == 0:
            self.set_category()
            return self
        # 找到最优特征及其对应的最优度量指标取值
        best_feature, best_metric = self.get_best_feature(tree_type=tree_type)
        # 度量指标小于阈值epsilon，返回
        if best_metric < epsilon:
            self.is_leaf = True
            return self
        self.feature = best_feature
        self.is_leaf = False
        self.set_category()
        self.available_features.remove(self.feature)
        # 根据特征A_g的取值，分割D为若干个非空子集
        for x in np.unique(self.train_x[:, best_feature]):
            new_x = self.train_x[self.train_x[:, best_feature] == x]
            new_y = self.train_y[self.train_x[:, best_feature] == x]
            # 构建子节点
            child = DecisionTree(
                x=new_x,
                y=new_y,
            )
            child.available_features = self.available_features
            child.parent = self
            self.children.update({x: child})
            child.fit(epsilon)

    def width_traversal(self):
        """决策树的宽度优先遍历"""
        q = Queue()
        q.put(self)
        height = 0
        end = self
        info = []
        while not q.empty():
            head = q.get()
            info.append((height, head))
            for value, child in head.children.items():
                q.put(child)
            if end == head:
                height += 1
                if not q.empty():
                    end = q.queue[-1]
        return info

    def show(self, feature_name: list = None):
        """绘制决策树"""
        if feature_name is None:
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
        for i, node in enumerate(node_info):
            mapper.update({node: points[i]})
            categories.append(node.category)
            cur_feature = feature_name[node.feature] if node.feature is not None else ""
            features.append(cur_feature)
        color = {x: i + 1 for i, x in enumerate(set(categories))}
        lines = []
        values = []
        for k, v in mapper.items():
            for feature_value, child in k.children.items():
                lines.append((v, mapper.get(child)))
                values.append(feature_value)
        for i, (x, y) in enumerate(points):
            plt.scatter(x, y, c=color_list[color[categories[i]]])
            plt.text(x, y - 5, categories[i])
            plt.text(x, y + 3, features[i])
        for i, (x, y) in enumerate(lines):
            dx = y[0] - x[0]
            dy = y[1] - x[1]
            plt.arrow(x[0], x[1], dx, dy)
            plt.text(x[0] + dx / 2, x[1] + dy / 2, values[i])
        plt.show()

    def get_best_feature(self, tree_type: TreeType = "ID3"):
        """获取最优分类特征"""
        best_feature = self.available_features[0]
        best_metrics = 0  # 度量标准的最优值
        for feature in self.available_features:
            if tree_type == "ID3":
                cur_metrics = self.cal_info_gain(feature)
            elif tree_type == "C45":
                cur_metrics = self.cal_info_gain_rate(feature)
            elif tree_type == "CART":
                cur_metrics = 0
                pass  # todo cart树
            else:
                raise ValueError("不支持的决策树类型")
            if cur_metrics > best_metrics:
                best_metrics = cur_metrics
                best_feature = feature
        return best_feature, best_metrics

    def predict(self, x: np.ndarray):
        """预测"""
        if self.is_leaf:
            return self.category
        else:
            children = self.children.get(x.flatten()[self.feature])
            return children.predict(x)

    def loss(self, alpha: float = 0):
        """计算决策树损失"""
        loss = 0
        leaves = 0
        if self.is_leaf:
            probs = self.get_classify_probs()
            entropy = self.cal_entropy(probs)
            loss += self.sample_num * entropy
            return loss, 1
        for _, child in self.children.items():
            if child.is_leaf:
                probs = child.get_classify_probs()
                entropy = child.cal_entropy(probs)
                loss += child.sample_num * entropy
                leaves += 1
            else:
                child_loss, child_leaves = child.loss()
                loss += child_loss
                leaves += child_leaves
        loss = loss + leaves * alpha
        return loss, leaves

    def root(self):
        res = self
        while res.parent is not None:
            res = res.parent
        return res

    def tree_loss(self, alpha: float):
        r = self.root()
        return r.loss(alpha)

    def pruning(self, alpha: float):
        """剪枝操作，避免过拟合"""
        if self.is_leaf:
            return
        children = list(self.children.values())
        # 优先处理非叶节点，这样可以优先去树的最深处进行剪枝
        children.sort(key=lambda x: x.is_leaf)
        for child in children:
            if not child.is_leaf:
                child.pruning(alpha=alpha)
            else:
                loss1, _ = child.tree_loss(alpha=alpha)
                child.parent.is_leaf = True
                loss2, _ = child.parent.tree_loss(alpha=alpha)
                if loss2 <= loss1:
                    child.parent.children = {}
                    child.parent.pruning(alpha)
                else:
                    child.parent.is_leaf = False


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
                ['老年', '否', '否', '一般', '否'],
                ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    df = np.array(datasets)
    train_x = df[:, :-1]
    train_y = df[:, -1]
    tree = DecisionTree(x=train_x, y=train_y)
    tree.fit(epsilon=0.1)
    # print(tree.loss(alpha=0.5))
    tree.show(labels)
    tree.pruning(alpha=1.9)
    tree.show(feature_name=labels)
    # print(tree.predict(np.array([["乌黑", "蜷缩", "浊响", "清晰", "凹陷", "硬滑"]])))
