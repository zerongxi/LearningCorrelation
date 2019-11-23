import numpy as np
from scipy.sparse import csr_matrix
import os
from sklearn.preprocessing import normalize
import pandas as pd

def edges2correlation_matrix(edges, weights=None):
    """
    :param edges: Nx2 numpy array
    :param weights: None or Nx(1) numpy array\
    :return:
        index2nodes
        correlation_matrix: 2D scipy sparse matrix

    """
    if weights is None:
        weights = np.ones((edges.shape[0], ), np.float32)
    edges = np.concatenate([edges, edges[:, [1, 0]]], axis=0)
    edges = np.swapaxes(edges, 0, 1)
    weights = np.tile(weights, (2,))
    correlation_matrix = csr_matrix((weights, edges), dtype=np.float32)
    if np.max(edges) <= 10**5:
        correlation_matrix = correlation_matrix.toarray()
    return correlation_matrix


def invert_index(src):
    dst = np.zeros((np.max(src)+1), np.int64)
    dst[src] = np.arange(src.size, dtype=np.int64)
    return dst


class Data:

    def __init__(self, folder_path):
        edges, node_group = self._load_csv(folder_path)
        self.index2node = np.unique(edges[:, [0, 1]])
        node2index = invert_index(self.index2node)
        weights = None if edges.shape[1] < 3 else edges[:, 2]
        self.correlation_matrix = \
            edges2correlation_matrix(node2index[edges[:, [0, 1]]], weights)
        node2group = np.zeros((np.max(node_group[:, 0])+1,), np.int16)
        node2group[node_group[:, 0]] = node_group[:, 1]
        appeared = np.isin(self.index2node, node_group[:, 0])
        self.correlation_matrix = self.correlation_matrix[appeared, :]
        self.correlation_matrix = self.correlation_matrix[:, appeared]
        self.index2node = self.index2node[appeared]
        self.index2group = node2group[self.index2node]
        return

    def _load_csv(self, folder_path):
        edges = pd.read_csv(
            os.path.join(folder_path, "edges.csv"),
            delimiter=",",
            dtype=np.int64,
            header=None
        ).values
        node_group = pd.read_csv(
            os.path.join(folder_path, "group-edges.csv"),
            delimiter=",",
            dtype=np.int64,
            header=None
        ).values
        return edges, node_group
