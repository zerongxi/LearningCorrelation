import numpy as np
from data import spc, synthetic


def split_data(data, ratio):

    n_samples = data.shape[0]
    split = [0]
    ret = []
    if type(ratio) is not list:
        ratio = [ratio, 1.0 - ratio]
    for cnt in range(len(ratio)):
        split.append(int(ratio[cnt] * n_samples))
        split[-1] += split[-2]
        ret.append(data[split[-2]:split[-1]])
    return ret


def frequency_filter(data, threshold):
    uniques, counts = np.unique(data, return_counts=True)
    index = counts >= threshold
    reserved = uniques[index]

    to_print = "Trace_length: %d, appeared_max: %d, appeared: %d, reserved_max: %d, reserved: %d, " \
               "cover addr: %.4f, cover access: %.4f" % (
        data.size, np.max(data), uniques.size, np.max(reserved), reserved.size,
        reserved.size / uniques.size, float(np.sum(counts[index]) / np.sum(counts))
    )
    print(to_print)

    return data[np.isin(data, reserved)]


def load_seq(seq_name):
    seq = synthetic.load(seq_name) if seq_name.lower().startswith("synthetic") else spc.load(seq_name)
    train, test = split_data(seq, 0.7)
    return train, test


def make_kernel(mode, window, max_diff=None):
    kernel = np.arange(1, window+1, dtype=np.float32)
    if mode == "constant":
        kernel = 1.0
    elif mode == "linear":
        kernel *= (max_diff / window)
    kernel /= np.sum(kernel)
    return kernel
