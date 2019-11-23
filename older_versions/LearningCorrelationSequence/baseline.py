import numpy as np
import torch


class LookupTable:

    def __init__(self, trace, k, kernel):
        self.index2addr = None
        self.index2kcorr = None
        self._learn(trace, k, kernel)
        return

    def _get_addr2index(self, index2addr):
        addr2index = -np.ones((np.max(index2addr)+1,), np.int64)
        addr2index[index2addr] = np.arange(index2addr.shape[0])
        return addr2index

    def _learn(self, trace, k, kernel):
        self.index2addr = np.unique(trace)
        trace = self._get_addr2index(self.index2addr)[trace]  # transform trace from addr to index
        self.index2kcorr = np.zeros((self.index2addr.shape[0], k), np.int64)
        for index in range(self.index2addr.shape[0]):
            table = np.zeros((self.index2addr.shape[0],), np.int64)
            positions = np.argwhere(trace == index)
            for pos in positions:
                beg = pos-kernel.shape[0]
                if beg < 0:
                    continue
                table[trace[beg:pos]] += kernel
            self.index2kcorr[index] = np.argsort(table)[:k]
        return

    def predict(self, trace, k):
        if k >= self.index2kcorr.shape[1]:
            ValueError("k is too large!")
        addr2index = self._get_addr2index(self.index2kcorr)

        valid = 0 <= trace <= np.max(self.index2addr)
        trace_indexed = -np.ones_like(trace)
        trace_indexed[valid] = addr2index[trace[valid]]
        valid = trace_indexed >= 0
        predict = -np.ones((trace_indexed.shape[0], k), np.int64)
        predict[valid] = self.index2kcorr[trace_indexed[valid]]
        predict[valid] = self.index2addr[predict[valid]]
        return predict


class RandomGuess:

    def __init__(self, data):
        self.addr_pool = np.unique(data)
        return

    def predict(self, data, k):
        predict_index = np.random.choice(self.addr_pool.size, data.shape[0] * k)
        predict_addr = self.addr_pool[predict_index]
        predict = predict_addr.reshape(-1, k)
        return predict


class RNN(torch.nn.Module):

    def __init__(self, in_dim, out_dim, embedding_dim, hidden_dim):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Embedding(in_dim, embedding_dim)
        self.lstm1 = torch.nn.LSTM(embedding_dim, hidden_dim)
        self.lstm2 = torch.nn.LSTM(hidden_dim, hidden_dim)
        self.linear = torch.nn.Linear(hidden_dim, out_dim)
        return

    def forward(self, data):
        embeds = self.embedding(data)
        lstm1_out, _ = self.lstm1(embeds.view(len(data), 1, -1))
        lstm2_out, _ = self.lstm2(lstm1_out.view(self.hidden_dim, 1, -1))
        result = torch.log_softmax(
            self.linear(lstm2_out.view(len(data), -1)),
            dim = 1
        )
        return result


class RecurrentNueralNetworks:

    def __init__(self, data):
        self.index2addr = np.unique(data)

        return

    def _learn(self):
        return

    def predict(self):
        return
