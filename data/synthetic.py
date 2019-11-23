import numpy as np
import pickle
from collections import namedtuple


SyntheticParameters = namedtuple(
    "SyntheticParameters",
    ["n_elements", "n_pairs", "seq_len", "n_agents",
     "agent_switch_probability", "agent_reinit_probability",
     "locality_probability", "locality_std"]
)


def generate_synthetic_dataset(parameters):
    n_elements = parameters["n_elements"]
    n_pairs = parameters["n_pairs"]

    # set correlation
    pairs = np.random.choice(n_elements, int(1.2 * 2 * n_pairs), replace=True)
    pairs = np.reshape(pairs, (-1, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]][:n_pairs]
    correlations = np.random.sample(n_pairs)

    # get correlated elements and their transition probabilities
    correlated_elements = [None] * n_elements
    transition_probabilities = [None] * n_elements
    for ele in range(n_elements):
        targets = np.logical_or(pairs[:, 0] == ele, pairs[:, 1] == ele)
        temp = pairs[targets]
        correlated_elements[ele] = np.where(temp[:, 1] == ele, temp[:, 0], temp[:, 1])
        transition_probabilities[ele] = correlations[targets]
        transition_probabilities[ele] /= np.sum(transition_probabilities[ele])
    return correlated_elements, transition_probabilities


def generate_synthetic_seq(parameters, correlated_elements=None, transition_probabilities=None):
    if correlated_elements is None or transition_probabilities is None:
        correlated_elements, transition_probabilities = generate_synthetic_dataset(parameters)

    agent_switch_probability = parameters["agent_switch_probability"]
    agent_reinit_probability = parameters["agent_reinit_probability"]
    locality_probability = parameters["locality_probability"]
    locality_std = parameters["locality_std"]

    n_elements = len(correlated_elements)
    seq_len = int(parameters["seq_len"])
    n_agents = int(parameters["n_agents"])
    seq = np.zeros((seq_len, ), np.int64)
    agents = np.random.choice(n_elements, n_agents)
    current_agent = 0
    for i in range(seq_len):
        if np.random.sample() < agent_switch_probability:
            current_agent = np.random.choice(n_agents)
        if np.random.sample() < agent_reinit_probability:
            agents[current_agent] = np.random.choice(n_elements)
        if np.random.sample() < locality_probability:
            seq[i] = min(n_elements, max(0, int(np.random.randn(1) * locality_std + agents[current_agent])))
        else:
            seq[i] = np.random.choice(correlated_elements[agents[current_agent]],
                                      p=transition_probabilities[agents[current_agent]])
        if (i + 1) % 100000 == 0:
            print("Generated {:8d}/{:8d}, Progress: {:.2f}".format(i+1, seq_len, i/seq_len))
    return seq


def load(seq_name):
    fpath = "./data/" + seq_name + ".pkl"
    with open(fpath, "rb") as f:
        seq = pickle.load(f)
    return seq


if __name__ == "__main__":
    n_elements = [5000, 10000, 20000, 50000]
    for i in range(len(n_elements)):
        parameters = SyntheticParameters(
            n_elements[i], 10*n_elements[i], 100*n_elements[i], 32,
            2e-1, 1e-2, 2e-1, 3.
        )._asdict()
        seq = generate_synthetic_seq(parameters)
        fpath = "./data/" + "synthetic_nelements_" + str(n_elements[i])+".pkl"
        with open(fpath, "wb") as f:
            pickle.dump(seq, f)
