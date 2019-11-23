import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from collections import namedtuple

from sequence_learners.sequence_learner import SeqLearner
import data.utils


TemporalMappingParameters = namedtuple(
    "TemporalMappingParameters",
    ["kernel_mode", "kernel_window", "kernel_diff",
     "alpha", "stop_criterion", "n_epochs", "n_dims", "norm",
     "beta", "min_repulse_distance", "n_neighbors", "gamma"]
)


class TemporalMapping(SeqLearner):

    def __init__(self, seq: np.ndarray, parameters: dict):
        super(TemporalMapping, self).__init__(seq, parameters)
        self.parameters["kernel"] = \
            data.utils.make_kernel(parameters["kernel_mode"], parameters["kernel_window"], parameters["kernel_diff"])
        self.index2vec = -1.0 + 2.0 * np.random.rand(self.index2ele.shape[0], self.parameters["n_dims"])
        ele2index = self._get_ele2index()
        seq = ele2index[seq]
        self._learn(seq)

        return

    def _learn(self, seq: np.ndarray):
        alpha = self.parameters["alpha"]
        beta = self.parameters["beta"]
        kernel = torch.from_numpy(self.parameters["kernel"]).double().unsqueeze(0).unsqueeze(1)\
            .cuda(self.parameters["cuda_id"])
        index2vec = torch.from_numpy(self.index2vec).cuda(self.parameters["cuda_id"])

        for epoch in range(self.parameters["n_epochs"]):
            updated = self._epoch(index2vec, seq, kernel, self.parameters["norm"], alpha, beta)
            diff = torch.mean(torch.sqrt(torch.sum(torch.pow(updated - index2vec, 2), dim=1)))
            index2vec = updated
            if (epoch + 1) % 20 == 0:
                print("Epcoh: {:4d}, diff: {:.3e}".format(epoch+1, diff))
            if diff < self.parameters["stop_criterion"]:
                print("Early stopping!")
                break

        self.index2vec = index2vec.cpu().numpy()
        return

    def _epoch(self, index2vec, seq, kernel, norm, alpha, beta):

        # attractice force
        seq_vec = index2vec[seq].permute(1, 0).unsqueeze(1)
        src = torch.conv1d(seq_vec, kernel).squeeze().permute(1, 0)
        tgt = torch.zeros_like(index2vec)
        tgt[seq[kernel.shape[-1]:]] += src[:-1]
        tgt /= torch.from_numpy(self.counts).double().unsqueeze(1).cuda(self.parameters["cuda_id"])
        updated = alpha * tgt + (1 - alpha) * index2vec

        # repulsive force
        if norm == "std":
            updated /= torch.std(updated, dim=0).unsqueeze(0)
            updated -= torch.mean(updated, dim=0).unsqueeze(0)
        elif norm == "force":
            n_neighbors = self.parameters["n_neighbors"]
            updated_np = updated.cpu().numpy()
            nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(updated_np)
            distance, neighbors = nbrs.kneighbors(updated_np)
            distance = distance[:, 1:]
            neighbors = neighbors[:, 1:]
            distance = torch.max(
                torch.from_numpy(np.array(self.parameters["min_repulse_distance"])).cuda(self.parameters["cuda_id"]),
                torch.from_numpy(distance).cuda(self.parameters["cuda_id"]))
            src = torch.pow(distance, self.parameters["gamma"])
            src = (updated.unsqueeze(1) -
                   updated[neighbors.flatten()].view((-1, n_neighbors, self.parameters["n_dims"]))) * \
                   src.unsqueeze(2)
            updated += beta * torch.sum(src, dim=1)
            print("Average distance of {}-nn: {:.3e}, std: {:.3e}".format(n_neighbors, torch.mean(distance), torch.mean(torch.std(updated, axis=0)).cpu().numpy()))

        return updated

    def predict(self, seq: np.ndarray, parameters: dict):
        ele2index = self._get_ele2index()
        valid = np.isin(seq, self.index2ele)
        seq_indexed = -np.ones_like(seq)
        seq_indexed[valid] = ele2index[seq[valid]]
        nbrs = NearestNeighbors(n_neighbors=parameters["top_k"]).fit(self.index2vec)
        _, neighbors = nbrs.kneighbors(self.index2vec)
        pred = -np.ones((seq_indexed.shape[0], parameters["top_k"]), np.int64)
        pred[valid] = neighbors[seq_indexed[valid]]
        pred[valid] = self.index2ele[pred[valid]]
        return pred
