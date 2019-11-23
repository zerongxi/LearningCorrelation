import numpy as np
import torch
from collections import namedtuple

from sequence_learners.sequence_learner import SeqLearner
from data.utils import make_kernel


CorrelationMatrixParameters = namedtuple(
    "CorrelationMatrixParameters",
    ["kernel_mode", "kernel_window", "kernel_diff"]
)


class CorrelationMatrix(SeqLearner):

    def __init__(self, seq: np.ndarray, parameters: dict):
        super(CorrelationMatrix, self).__init__(seq, parameters)

        ele2index = self._get_ele2index()
        seq = ele2index[seq]
        self.index2kcorr = None
        self._learn(seq)
        return

    def _learn(self, seq: np.ndarray):
        kernel = make_kernel(mode=self.parameters["kernel_mode"], window=self.parameters["kernel_window"],
                             max_diff=self.parameters["kernel_diff"])
        kernel = torch.from_numpy(kernel).cuda(self.parameters["cuda_id"])
        seq = torch.from_numpy(seq).cuda(self.parameters["cuda_id"])
        self.index2kcorr = np.zeros((self.index2ele.shape[0], self.parameters["top_k"]), np.int64)
        for index in range(self.index2kcorr.shape[0]):
            corr = torch.zeros((self.index2kcorr.shape[0], ), dtype=torch.float32)\
                .cuda(self.parameters["cuda_id"])
            positions = (seq == index).nonzero()
            for pos in positions:
                beg = pos - kernel.shape[0]
                if beg < 0:
                    continue
                corr[seq[beg:pos]] += kernel
            self.index2kcorr[index] =\
                torch.argsort(corr, descending=True)[:self.parameters["top_k"]].cpu().numpy()
        return

    def predict(self, seq: np.ndarray, parameters):
        if parameters["top_k"] > self.index2kcorr.shape[1]:
            ValueError("k is too large!")
        ele2index = self._get_ele2index()

        valid = (0 <= seq) & (seq <= np.max(self.index2ele))
        seq_indexed = -np.ones_like(seq)
        seq_indexed[valid] = ele2index[seq[valid]]
        valid = seq_indexed >= 0
        predict = -np.ones((seq_indexed.shape[0], parameters["top_k"]), np.int64)
        predict[valid] = self.index2kcorr[seq_indexed[valid]]
        predict[valid] = self.index2ele[predict[valid]]
        return predict

