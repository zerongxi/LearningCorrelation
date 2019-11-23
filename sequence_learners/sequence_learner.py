import numpy as np
from collections import namedtuple


GeneralParameters = namedtuple(
    "GeneralParameters",
    ["top_k", "freq_threshold"]
)


class SeqLearner:

    def __init__(self, seq: np.ndarray, parameters: dict):
        self.parameters = parameters
        self.index2ele, self.counts = np.unique(seq, return_counts=True)
        return

    def _get_ele2index(self):
        ele2index = -np.ones((np.max(self.index2ele)+1,), np.int64)
        ele2index[self.index2ele] = np.arange(self.index2ele.shape[0])
        return ele2index

    def _learn(self, seq: np.ndarray):
        return

    def predict(self, seq, parameters):
        return
