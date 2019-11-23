import torch
import numpy as np
from collections import namedtuple
import pickle, os

from sequence_learners.sequence_learner import SeqLearner
from sequence_learners.temporal_mapping import TemporalMapping


MutualMappingParameters = namedtuple(
    "MutualMappingParameters",
    ["tmap", "n_epochs", "hidden_dim", "learning_rate", "batch_size"]
)


class NN(torch.nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim):
        super(NN, self).__init__()
        self.linear1 = torch.nn.Linear(in_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, out_dim)
        return

    def forward(self, data):
        x = self.linear1(data)
        x.relu()
        x = self.linear2(x)
        x.relu()
        x = self.linear3(x)
        x.relu()
        return x


class DataLoader:

    def __init__(self, binary, vec, parameters):
        self.batch_size = parameters["batch_size"]
        self.binary = binary
        self.vec = vec
        self.order = None
        self.current = None
        self._prepare()
        return

    def _prepare(self):
        self.order = np.random.permutation(self.binary.shape[0])
        self.current = 0
        return

    def get_batch(self):
        if self.batch_size + self.current > self.order.size:
            self._prepare()
            return None, None
        batch = self.order[self.current:self.current+self.batch_size]
        self.current += self.batch_size
        return self.binary[batch], self.vec[batch]


def int2binary(numbers, n_bits=32):
    numbers = np.array(numbers)
    shape = numbers.shape
    binaries = map(lambda n: np.array(list(np.binary_repr(n).zfill(n_bits)), dtype=np.uint8), numbers.flatten())
    binaries = np.stack(list(binaries), axis=0)
    binaries.reshape(*shape, -1)
    return binaries


def binary2int(binaries):
    kernel = np.power(2, np.arange(binaries.shape[-1]))[::-1]
    numbers = np.sum(binaries * kernel, axis=-1)
    return numbers


class MutualMapping(SeqLearner):

    def __init__(self, seq: np.ndarray, parameters: dict):
        super(MutualMapping, self).__init__(seq, parameters)
        self.parameters["tmap"]["cuda_id"] = self.parameters["cuda_id"]
        self.binary2vec = None
        self.vec2binary = None
        self.log_path = self.parameters["result_path"][:-4]+".txt"
        self._learn(seq)
        return

    def _log(self, content):
        with open(self.log_path, "a+") as f:
            f.write(content)
        return

    def _learn(self, seq):
        temp_path = "tmap.pkl"
        if os.path.exists(temp_path):
            with open(temp_path, "rb") as f:
                tmap = pickle.load(f)
        else:
            tmap = TemporalMapping(seq, self.parameters["tmap"])
            with open(temp_path, "wb") as f:
                pickle.dump(tmap, f)
        self.index2ele = tmap.index2ele
        index2binary = int2binary(tmap.index2ele)
        self.n_bits = index2binary.shape[1]
        index2binary = torch.from_numpy(index2binary.astype(np.float32))
        index2vec = torch.from_numpy(tmap.index2vec.astype(np.float32))
        dataloader = DataLoader(index2binary, index2vec, self.parameters)
        cuda_id = self.parameters["cuda_id"]

        # train binary2vec
        self._log("\nStart to train binary2vec...")
        self.binary2vec = NN(index2binary.shape[1], index2vec.shape[1], self.parameters["hidden_dim"])
        self.binary2vec = self.binary2vec.cuda(cuda_id).train()
        optimizer = torch.optim.Adam(self.binary2vec.parameters(), self.parameters["learning_rate"])
        loss_fn = torch.nn.MSELoss()
        best = [100., 0]
        for epoch in range(self.parameters["n_epochs"]):
            binary, vec = dataloader.get_batch()
            total_loss, total_count = 0., 0
            while binary is not None and vec is not None:
                binary = binary.cuda(cuda_id)
                vec = vec.cuda(cuda_id)
                self.binary2vec.zero_grad()
                pred = self.binary2vec(binary)
                loss = loss_fn(pred.flatten(), vec.flatten())
                loss.backward()
                optimizer.step()
                total_count += 1
                total_loss += loss.cpu().detach().numpy()
                binary, vec = dataloader.get_batch()
            self._log("Epoch:{:3d}, avg_loss:{:.6e}\n".format(epoch, total_loss / total_count))
            if total_loss > best[0]:
                if epoch - best[1] > 100:
                    self._log("Early stopping!")
                    break
            else:
                best = [total_loss, epoch]
        self.binary2vec = self.binary2vec.cpu()

        # train vec2binary
        self._log("\nStart to train vec2binary...")
        self.vec2binary = NN(index2vec.shape[1], index2binary.shape[1], self.parameters["hidden_dim"])
        self.vec2binary = self.vec2binary.cuda(cuda_id).train()
        optimizer = torch.optim.Adam(self.vec2binary.parameters(), self.parameters["learning_rate"])
        loss_fn = torch.nn.BCEWithLogitsLoss()
        best = [100., 0]
        for epoch in range(self.parameters["n_epochs"]):
            binary, vec = dataloader.get_batch()
            total_loss, total_count = 0., 0
            while binary is not None and vec is not None:
                binary = binary.cuda(cuda_id)
                vec = vec.cuda(cuda_id)
                self.vec2binary.zero_grad()
                pred = self.vec2binary(vec)
                loss = loss_fn(pred, binary)
                loss.backward()
                optimizer.step()
                total_count += 1
                total_loss += loss.cpu().detach().numpy()
                total_loss /= total_count
                binary, vec = dataloader.get_batch()
            self._log("Epoch:{:3d}, avg_loss:{:.6e}\n".format(epoch, total_loss))
            if total_loss > best[0]:
                if epoch - best[1] > 100:
                    self._log("Early stopping!")
                    break
            else:
                best = [total_loss, epoch]
        self.vec2binary = self.vec2binary.cpu()
        return

    def predict(self, seq, parameters):
        cuda_id = self.parameters["cuda_id"]
        binary = int2binary(seq)
        binary2vec = self.binary2vec.cuda(cuda_id).eval()
        vec2binary = self.vec2binary.cuda(cuda_id).eval()
        batch_size = 1024
        n_batches = int(np.ceil(seq.shape[0] / batch_size))
        pred = []
        for batch in range(n_batches):
            binary_batch = torch.from_numpy(binary[batch*batch_size:(batch+1)*batch_size].astype(np.float32)).cuda(cuda_id)
            vec_batch = binary2vec(binary_batch)
            vec_neighbors = vec_batch.unsqueeze(1) + \
                torch.randn(vec_batch.shape[0], parameters["top_k"], vec_batch.shape[1], dtype=torch.float32).cuda(cuda_id)
            binary_neighbors = (torch.sigmoid(vec2binary(vec_neighbors.view(-1, vec_batch.shape[-1]))) > 0.5)
            binary_neighbors = binary_neighbors.detach().cpu().numpy()
            pred.append(binary2int(binary_neighbors))
            pred[-1] = pred[-1].reshape((-1, parameters["top_k"]))
        pred = np.concatenate(pred, axis=0)
        return pred
