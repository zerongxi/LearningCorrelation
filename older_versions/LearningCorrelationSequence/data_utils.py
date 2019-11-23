import numpy as np


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
