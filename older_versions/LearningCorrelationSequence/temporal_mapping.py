import numpy as np
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
import synthetic_data
import spc_data
import data_utils
import eval_utils


N_EPOCHES = 2000
STOP_CRITERION = 1e-5
RESULT_PATH = "./results/"


class TemporalMapping:

    def __init__(self, trace, kernel, model_path=None, norm="std", step=False):

        self.index2addr, self.counts = np.unique(trace, return_counts=True)
        self.index2pt = -1.0 + 2.0 * np.random.rand(self.index2addr.size, kernel.shape[1])
        self.knn = None

        self._learn(trace, kernel, norm, step)
        # del self.counts
        return

    def _expand(self, index2pt):
        addr2pt = np.zeros((np.max(self.index2addr)+1, index2pt.shape[1]), np.float32)
        addr2pt[self.index2addr] = index2pt
        return addr2pt

    def _squeeze(self, addr2pt):
        index2pt = addr2pt[self.index2addr]
        return index2pt

    def _learn(self, trace, kernel, norm, step, n_epoches=N_EPOCHES, stop_criterion=STOP_CRITERION):
        alpha = 2e-1
        beta = 5e-5
        start_t = datetime.now()
        for epoch in range(n_epoches):
            updated = self._epoch(trace, kernel, norm, step, alpha, beta)
            updated -= np.average(updated, axis=0)
            diff = np.average(np.sqrt(np.sum(np.square(updated - self.index2pt), axis=1)))

            self.index2pt = updated
            if epoch % 10 == 9:
                print(
                    "epoch {:3d}, diff: {:.2e}, range: ({:.1f}, {:.1f}) time {}".format(epoch+1, diff, np.max(updated), np.min(updated), datetime.now()-start_t)
                )
            if diff < stop_criterion:
                break
            # alpha *= 0.995
            # beta *= 0.99

        return

    def _epoch(self, trace, kernel, norm, step, alpha, beta):
        addr2pt = self._expand(self.index2pt)
        kernel = kernel[::-1]

        if step:
            pass
        else:
            # attractive force
            src = np.stack([
                np.convolve(addr2pt[trace, i], kernel[:, i])[kernel.shape[0]-1:-kernel.shape[0]-1]
                for i in range(kernel.shape[1])
            ], axis=1)
            tgt = np.zeros_like(addr2pt)
            tgt[trace[kernel.shape[0]+1:]] += src
            updated = self._squeeze(tgt)
            updated /= np.expand_dims(self.counts, axis=1)
            updated *= alpha
            updated += (1 - alpha) * self.index2pt

            # repulsive force
            if norm == "force":
                n_candidates = 100
                nbrs = NearestNeighbors(n_neighbors=n_candidates).fit(self.index2pt)
                distance, neighbors = nbrs.kneighbors(self.index2pt)
                distance = np.maximum(1e-3, distance)
                src = np.power(distance, -2)
                src = (
                              np.expand_dims(self.index2pt, axis=1)
                              - self.index2pt[neighbors.flatten()].reshape((-1, n_candidates, self.index2pt.shape[1]))
                      ) * np.expand_dims(src, axis=2)
                updated += beta * np.sum(src, axis=1)
                print("average distance of 100-nn: {:.6f}, deviation: {}"
                      .format(np.average(distance), np.std(updated, axis=0)))
            elif norm == "std":
                stds = np.std(updated, axis=0)
                updated /= stds
                # print("std: {}".format(stds))
        return updated

    def predict(self, trace, k):
        nbrs = NearestNeighbors(n_neighbors=k).fit(self.index2pt)
        addr2pt = self._expand(self.index2pt)
        _, neighbors = nbrs.kneighbors(addr2pt[trace])
        return self.index2addr[neighbors.flatten()].reshape(-1, k)


def make_kernel(modes, window, max_diffs=None, n_dims=None):
    if type(modes) != list:
        if type(max_diffs) != list:
            modes = [modes for _ in range(n_dims)]
            max_diffs = [max_diffs for _ in range(n_dims)]
        else:
            modes = [modes for _ in range(len(max_diffs))]
    if type(max_diffs) != list:
        max_diffs = [max_diffs for _ in range(len(modes))]
    kernel = np.stack([np.arange(1, window+1, dtype=np.float32) for _ in range(len(modes))], axis=1)
    for i in range(len(modes)):
        if modes[i] == "constant":
            kernel[i, :] = 1.0
        elif modes[i] == "step":
            kernel[i] *= (max_diffs[i] / window)
    kernel /= np.sum(kernel, axis=0)
    return kernel


def tmapping_experiment(trace, threshold, norm, kernel_window, kernel_modes, kernel_diffs, k=10, window=128):
    if trace.lower() == "synthetic":
        data = synthetic_data.load_data()
    else:
        data = spc_data.load_data(trace)
    data = data_utils.frequency_filter(data, threshold)
    train, test = data_utils.split_data(data, 0.7)
    kernel = make_kernel(kernel_modes, kernel_window, kernel_diffs)

    tmapping = TemporalMapping(train, kernel, norm=norm)

    name = "{}_threshold_{}_ndims_{}".format(trace, threshold, kernel.shape[1])

    # eval train data
    pred = tmapping.predict(train, k)
    accu = eval_utils.calc_accuracy(train[1:], pred[:-1], window)
    fpath = RESULT_PATH + "tmapping_" + name + "_train.csv"
    eval_utils.save_accuracy(fpath, accu, k, window)

    # eval test data
    pred = tmapping.predict(test, k)
    accu = eval_utils.calc_accuracy(test[1:], pred[:-1], window)
    fpath = RESULT_PATH + "tmapping_" + name + "_test.csv"
    eval_utils.save_accuracy(fpath, accu, k, window)
    return


if __name__ == "__main__":
    for file in ["synthetic", "Financial1", "Financial2", "WebSearch1", "WebSearch2", "WebSearch3"]:
        tmapping_experiment(file, 10, "std", 32, "step", [3, 6, 10])
