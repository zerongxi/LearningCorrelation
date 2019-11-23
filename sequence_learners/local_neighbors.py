import numpy as np
from collections import namedtuple

from sequence_learners.sequence_learner import SeqLearner


LocalNeighborsParameters = namedtuple(
    "LocalNeighborsParameters",
    ["next_probability"]
)


class LocalNeighbors(SeqLearner):

    def __init__(self, seq, parameters):
        super(LocalNeighbors, self).__init__(seq, parameters)

        return

    def predict(self, seq: np.ndarray, parameters: dict):
        choice = np.random.sample(seq.shape[0]) < parameters["next_probability"]
        pred = np.where(choice,
                        [seq+i+1 for i in range(parameters["top_k"])], [seq-i-1 for i in range(parameters["top_k"])])
        pred = np.swapaxes(pred, 0, 1)
        return pred
