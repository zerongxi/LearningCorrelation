import numpy as np
import pickle
import os


SYNTHETIC_DATAPATH = "./data/synthetic.pkl"


class SyntheticData:

    def __init__(self, data_range, n_correlated_pairs, max_correlation):
        self.range = int(data_range)
        self.pairs = None
        self.correlations = None
        self._set_correlation(int(n_correlated_pairs), max_correlation)
        pass

    def _set_correlation(self, n_pairs, max_correlation):
        pairs = np.random.choice(self.range, int(1.1 * n_pairs), replace=True)
        pairs = np.reshape(pairs, (-1, 2))
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        pairs = pairs[:n_pairs]
        correlations = np.random.sample(n_pairs)
        correlations *= max_correlation

        self.pairs = pairs
        self.correlations = correlations
        return

    def _get_indexed_correlation(self):
        _, counts = np.unique(self.pairs, return_counts=True)
        positions = np.zeros((self.range, np.max(counts)+1), np.int64)
        probabilities = np.zeros_like(positions, np.float32)
        positions[:, 0] = np.arange(self.range)
        probabilities[:, 0] = 1.0
        current = np.ones((self.range, ), np.int64)
        for i in range(self.pairs.shape[0]):
            a, b = self.pairs[i]
            positions[a, current[a]] = b
            probabilities[a, current[a]] = self.correlations[i]
            current[a] += 1
            positions[b, current[b]] = a
            probabilities[b, current[b]] = self.correlations[i]
            current[b] += 1
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        return positions, probabilities

    def sample(self, mode, n_trace, n_threads, thread_transit_probability,
               position_transit_probability, std):
        n_trace = int(n_trace)
        n_threads = int(n_threads)
        dtypes = dict(discrete=np.int64, continuous=np.float32)
        trace = np.zeros((n_trace, ), dtypes[mode])
        threads = np.random.choice(self.range, n_trace)
        current = 0
        positions, probabilities = self._get_indexed_correlation()
        for i in range(n_trace):
            if np.random.sample() < thread_transit_probability:
                current = np.random.choice(n_threads)
            if np.random.sample() < position_transit_probability:
                threads[current] = np.random.choice(self.range)
            position = np.random.choice(positions[threads[current]], p=probabilities[threads[current]])
            trace[i] = min(self.range, max(1, np.random.randn(1) * std + position))
            if i % 10**5 == 0:
                print(i/n_trace)
        return trace


def load_data(fpath=SYNTHETIC_DATAPATH, data_range=1e5, n_correlated_pairs=1e6, correlations=0.1,
              mode="discrete", n_trace=1e7, n_threads=50, thread_transit_probability=0.1,
              position_transit_probability=0.1, std=3.0):
    if os.path.exists(fpath):
        with open(fpath, "rb") as f:
            data = pickle.load(f)
    else:
        generator = SyntheticData(data_range, n_correlated_pairs, correlations)
        data = generator.sample(mode, n_trace, n_threads, thread_transit_probability,
                                 position_transit_probability, std)
        with open(fpath, "wb") as f:
            pickle.dump(data, f)
    data = data[data < (100 * np.max(data))]
    return data


if __name__ == "__main__":
    load_data()
