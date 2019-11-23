import numpy as np
import multiprocessing as mp
from pool_utils import put_array_into_shared_memory, get_array_from_shared_memory
import torch


def calc_accuracy_worker(trace, pred, accu_mat, w, k, lock):
    trace = get_array_from_shared_memory(*trace)
    pred = get_array_from_shared_memory(*pred)
    accu_mat = get_array_from_shared_memory(*accu_mat)
    accu_mat[w, k] = np.average(np.equal(trace[w:-1], pred[:-w-1, k]))
    lock.release()
    return


def calc_accuracy_parallel(trace, pred, window):
    trace = trace.astype(np.int64)
    pred = pred.astype(np.int64)
    accu_mat = np.zeros((window, pred.shape[1]), np.float32)
    shared_trace = put_array_into_shared_memory(trace, False)
    shared_pred = put_array_into_shared_memory(pred, False)
    shared_accu_mat = put_array_into_shared_memory(accu_mat, False)

    semaphore = mp.Semaphore(16)
    processes = []
    for w in range(accu_mat.shape[0]):
        for k in range(accu_mat.shape[1]):
            semaphore.acquire()
            p = mp.Process(
                target=calc_accuracy_worker,
                args=(shared_trace, shared_pred, shared_accu_mat, w, k, semaphore)
            )
            processes.append(p)
            p.start()
    for p in processes:
        p.join()

    ret = np.zeros_like(accu_mat)
    for w in range(ret.shape[0]):
        for k in range(ret.shape[1]):
            ret[w, k] = np.sum(accu_mat[:w+1, :k+1])

    return ret


def calc_accuracy(data, pred, window):
    data = data.astype(np.int64)
    pred = pred.astype(np.int64)
    accu_mat = np.zeros((window, pred.shape[1]), np.float32)
    for w in range(accu_mat.shape[0]):
        for k in range(accu_mat.shape[1]):
            accu_mat[w, k] = np.average(np.equal(data[w:-1], pred[:-w-1, k]))

    ret = np.zeros_like(accu_mat)
    for w in range(ret.shape[0]):
        for k in range(ret.shape[1]):
            ret[w, k] = np.sum(accu_mat[:w+1, :k+1])
    return ret


def save_accuracy(fpath, accu, topk, window):
    k_sample = np.array([1, 3, 5, 10, 20]) - 1
    window_sample = np.array([1, 5, 10, 20, 50, 100]) - 1
    k_sample = k_sample[k_sample < topk]
    window_sample = window_sample[window_sample < window]
    buffer = "k \ window," + ",".join([str(ele+1) for ele in window_sample]) + "\n"
    for k in k_sample:
        buffer += str(k+1) + "," + ",".join([str(ele) for ele in accu[window_sample, k]]) + "\n"
    with open(fpath, "w") as f:
        f.write(buffer)
    return


def calc_accuracy_gpu(data, pred, window):
    data = torch.from_numpy(data.astype(np.int64)).cuda()
    pred = torch.from_numpy(pred.astype(np.int64)).cuda()
    accu_mat = torch.zeros((window, pred.shape[1]), dtype=torch.float32).cuda()
    for w in range(accu_mat.shape[0]):
        for k in range(accu_mat.shape[1]):
            accu_mat[w, k] = torch.mean(torch.equal(data[w:-1], pred[:-w-1, k]))
    accu_mat = accu_mat.cpu().numpy()

    ret = np.zeros_like(accu_mat)
    for w in range(ret.shape[0]):
        for k in range(ret.shape[1]):
            ret[w, k] = np.sum(accu_mat[:w+1, :k+1])
    return ret
