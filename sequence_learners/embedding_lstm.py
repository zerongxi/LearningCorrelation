import torch
import numpy as np
from collections import namedtuple

from sequence_learners.sequence_learner import SeqLearner


EmbeddingLSTMParameters = namedtuple(
    "RNNParameters",
    ["embedding_dim", "hidden_dim", "sentence_len", "step_len", "batch_size", "n_epochs", "freq_threshold",
     "learning_rate", "max_ele", "n_steps"]
)


class Net(torch.nn.Module):

    def __init__(self, in_dim, out_dim, embedding_dim, hidden_dim):
        super(Net, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Embedding(in_dim, embedding_dim)
        self.lstm1 = torch.nn.LSTM(embedding_dim, hidden_dim)
        self.lstm2 = torch.nn.LSTM(hidden_dim, hidden_dim)
        self.linear = torch.nn.Linear(hidden_dim, out_dim)
        return

    def forward(self, data):
        embeds = self.embedding(data)
        lstm1_out, _ = self.lstm1(embeds)
        lstm2_out, _ = self.lstm2(lstm1_out)
        result = self.linear(lstm2_out)
        return result


class DataLoader:

    def __init__(self, seq: np.ndarray, parameters: dict):
        self.seq = torch.from_numpy(seq)
        self.n_ele = np.unique(seq).shape[0]
        self.parameters = parameters
        self._prepare()
        return

    def _prepare(self):
        sentence_len = self.parameters["sentence_len"]
        self.n_sentences = int((self.seq.shape[0] - sentence_len) / self.parameters["step_len"] + 1)
        self.n_batches = int(self.n_sentences / self.parameters["batch_size"])
        self.order = np.random.permutation(np.arange(self.n_sentences))
        self.current = 0

    def get_batch(self):
        sentence_len = self.parameters["sentence_len"]
        step_len = self.parameters["step_len"]
        batch_size = self.parameters["batch_size"]

        # if data is exhausted, re-prepare data and return None
        if self.current + batch_size > self.n_sentences:
            self._prepare()
            return None

        batch = torch.zeros((sentence_len, batch_size), dtype=torch.int64)
        for i in range(batch_size):
            beg = self.order[i+self.current] * step_len
            end = beg + sentence_len
            batch[:, i] = self.seq[beg:end]
        self.current += batch_size
        return batch


class EmbeddingLSTM(SeqLearner):

    def __init__(self, seq: np.ndarray, parameters: dict):
        super(EmbeddingLSTM, self).__init__(seq, parameters)
        self.log_path = self.parameters["result_path"][:-4]+".txt"
        del self.index2ele, self.counts
        seq = seq[1:] - seq[:-1]
        self.index2ele, self.counts = np.unique(seq, return_counts=True)
        reserved = np.argsort(self.counts)[::-1][:parameters["max_ele"]]
        self.counts = self.counts[reserved]
        self.index2ele = self.index2ele[reserved]
        seq = seq[np.isin(seq, self.index2ele)]

        self.shift = np.min(seq)
        seq -= self.shift
        self.index2ele -= self.shift

        n_ele = self.index2ele.shape[0]
        self.net = Net(n_ele, n_ele, self.parameters["embedding_dim"], self.parameters["hidden_dim"])\
            .cuda(self.parameters["cuda_id"])
        ele2index = self._get_ele2index()
        self._learn(ele2index[seq])
        return

    def _log(self, content):
        with open(self.log_path, "a+") as f:
            f.write(content)
        return

    def _learn(self, seq: np.ndarray):
        self.net = self.net.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.parameters["learning_rate"])
        dataloader = DataLoader(seq, self.parameters)
        least_loss = [100., -1]
        n_epochs = int(self.parameters["n_steps"] / dataloader.n_batches)
        print("N_EPOCHS: {}".format(n_epochs))
        for epoch in range(n_epochs):
            data = dataloader.get_batch()
            total_loss, total_count = 0., 0
            while data is not None:
                data = data.cuda(self.parameters["cuda_id"])
                self.net.zero_grad()
                pred = self.net(data[:-1])
                loss = loss_fn(pred.view(-1, pred.shape[-1]), data[1:].flatten())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.)
                optimizer.step()
                total_count += 1
                total_loss += loss.cpu().detach().numpy()
                data = dataloader.get_batch()
            total_loss /= total_count
            self._log("Epoch:{:3d}, avg_loss:{:.6e}\n".format(epoch, total_loss))
            if total_loss > least_loss[0]:
                if epoch - least_loss[1] > 10:
                    self._log("Early stopping!")
                    break
            else:
                least_loss[0] = total_loss
                least_loss[1] = epoch
        return

    def predict(self, seq: np.ndarray, parameters: dict):
        seq_addr = np.copy(seq)
        seq = np.concatenate([[0], seq], axis=0)
        seq = seq[1:] - seq[:-1]
        seq = seq - self.shift
        valid = np.isin(seq, self.index2ele)
        ele2index = self._get_ele2index()
        seq_valid = ele2index[seq[valid]]
        data = torch.from_numpy(seq_valid).view(-1, 1)
        net = self.net.eval()
        sentence_len = int(parameters["batch_size"] * parameters["sentence_len"] / 2)
        results = []
        for i in range(int(np.ceil(data.shape[0] / sentence_len))):
            results.append(
                torch.argsort(
                    net(data[i * sentence_len:(i + 1) * sentence_len].cuda(self.parameters["cuda_id"])).squeeze().detach(),
                    dim=-1,
                    descending=True
                )[:, :parameters["top_k"]].cpu().numpy()
            )
        pred = -1000 * np.ones((seq.shape[0], parameters["top_k"]), np.int64)
        pred[valid] = np.concatenate(results)
        pred[valid] = self.index2ele[pred[valid]]
        pred[valid] += np.expand_dims(seq_addr[valid], 1) + self.shift
        return pred




