#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File: knn.py
@Time: 2023/2/23-16:30
@Author: Li Dongchao
@Desc:
@release:
"""

from typing import Union
import numpy as np


class MinimumList(list):
    """最小值列表"""

    def __init__(self, k: int):
        if k < 1:
            raise ValueError(f"Param k except bigger than or equal to 1, but get '{k}'")
        super().__init__()
        self.k = k

    def full(self):
        return self.__len__() == self.k

    def append(self, value: tuple):
        if self.__len__() < self.k:
            super().append(value)
            self.sort(key=lambda x: x[0])
        else:
            if value[0] < self[-1][0]:
                self.pop()
                super().append(value)
                self.sort(key=lambda x: x[0])

    def get_nodes(self):
        return [item[1] for item in self]

    def contains_node(self, node):
        """包含node节点"""
        return node in self.get_nodes()

    def max(self):
        return self[-1][0]

    def min(self):
        return self[0][0]


class Node(object):

    def __init__(
            self,
            *,
            data: Union[np.ndarray, None] = None,
            dim: Union[int, None] = None,
            left=None,
            right=None
    ):
        self.data = data
        self.dim = dim
        self.left: Node = left
        self.right: Node = right
        self.parent: Union[Node, None] = None
        self.other: Union[Node, None] = None

    def __repr__(self):
        return self.data.__str__()

    @staticmethod
    def create(train_data: np.ndarray, cur_dim: int = 0):
        """生成kd数"""
        n_sample, n_feature = train_data.shape
        if n_sample <= 0:
            return
        train_data = train_data[train_data[:, cur_dim].argsort()]
        arg_median = int(n_sample / 2)
        median_data = train_data[arg_median, :]
        new_train_data = np.delete(train_data, arg_median, axis=0)
        left_data = new_train_data[new_train_data[:, cur_dim] < median_data[cur_dim]]
        right_data = new_train_data[new_train_data[:, cur_dim] >= median_data[cur_dim]]
        new_dim = (cur_dim + 1) % n_feature
        left_node = Node.create(left_data, cur_dim=new_dim)
        right_node = Node.create(right_data, cur_dim=new_dim)
        if left_node is not None:
            left_node.other = right_node
        if right_node is not None:
            right_node.other = left_node
        node = Node(data=median_data, dim=cur_dim, left=left_node, right=right_node)
        if left_node is not None:
            left_node.parent = node
        if right_node is not None:
            right_node.parent = node
        return node

    @staticmethod
    def _median(data: np.ndarray):
        """计算一组数的中位数"""
        median = np.median(data)
        return data[data > median].min()

    def traversal(self):
        """先序遍历"""
        print(self.data)
        if self.left is not None:
            self.left.traversal()
        if self.right is not None:
            self.right.traversal()

    def find(self, x: np.ndarray, k: int):
        """查找k近邻"""
        k_neighbor = MinimumList(k)
        leaf = self._go_to_leaf(x=x, k_neighbor=k_neighbor)
        leaf.backtrack(x, k_neighbor, stop_node=Node())
        return k_neighbor

    def backtrack(self, x: np.ndarray, k_neighbor: MinimumList, stop_node):
        """回溯"""
        if self.parent is None:  # 到达根节点，结束
            return k_neighbor
        if not k_neighbor.contains_node(self.parent):
            d = self.distance(x, self.parent.data)
            k_neighbor.append((d, self.parent))
        if self.other == stop_node:
            return k_neighbor
        d1 = abs(x[self.parent.dim] - self.parent.data[self.parent.dim])  # 目标点到父节点分割线（超平面）的距离
        if d1 < k_neighbor.max() or not k_neighbor.full():
            if self.other is not None:
                leaf = self.other._go_to_leaf(x, k_neighbor)
                leaf.backtrack(x, k_neighbor, stop_node=self)
                return k_neighbor
        self.parent.backtrack(x, k_neighbor, stop_node=Node())

    def _go_to_leaf(self, x: np.ndarray, k_neighbor: MinimumList):
        """二叉查找，找到叶节点"""
        if x[self.dim] < self.data[self.dim]:
            if self.left is None:
                d = self.distance(x, self.data)
                k_neighbor.append((d, self))
                return self
            return self.left._go_to_leaf(x, k_neighbor)
        else:
            if self.right is None:
                d = self.distance(x, self.data)
                k_neighbor.append((d, self))
                return self
            return self.right._go_to_leaf(x, k_neighbor)

    @staticmethod
    def distance(x: np.ndarray, y: np.ndarray, p: int = 2):
        """计算两点之间的LP距离"""
        if p < 1:
            raise ValueError(f"Params p expect greater than or equal to 1, but get '{p}'")
        d = np.power(np.abs(x - y), p).sum() ** (1.0 / p)
        return d


if __name__ == '__main__':
    mydata = np.array([(2, 3), (7, 4), (9, 6), (4, 7), (8, 1), (7, 2)])
    kd_tree = Node.create(mydata)
    kd_tree.traversal()
    zz = kd_tree.find(x=np.array([2.1, 3.1]), k=2)
    print(zz)
