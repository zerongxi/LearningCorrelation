import numpy as np

from sequence_learners.sequence_learner import SeqLearner


class InformedGuess(SeqLearner):

    def __init__(self, seq: np.ndarray, parameters: dict):
        super(InformedGuess, self).__init__(seq, parameters)

        self.prior = self.counts / np.sum(self.counts)
        return

    def predict(self, seq, parameters: dict):
        pred = np.random.choice(
            self.index2ele.shape[0], seq.shape[0] * parameters["top_k"],
            p=self.prior,
            replace=True)
        pred = self.index2ele[pred].reshape((seq.shape[0], -1))
        return pred
