import numpy as np
import torch


def calc_accuracy(pred: np.ndarray, target: np.ndarray, window, mode, cuda_id=0):
    pred = torch.from_numpy(pred).cuda(cuda_id)
    target = torch.from_numpy(target).cuda(cuda_id)
    result = torch.zeros((window, pred.shape[1]), dtype=torch.float32)
    for k in range(result.shape[1]):
        hit = torch.zeros((target.shape[0]-window+1,),
                          dtype=torch.float32 if mode == "occurrence" else torch.bool).cuda(cuda_id)
        for w in range(result.shape[0]):
            if mode == "occurrence":
                hit += torch.eq(pred[:-window+1, k], target[w:target.shape[0]-window+1+w]).float()
            else:
                hit |= torch.eq(pred[:-window+1, k], target[w:target.shape[0]-window+1+w])
            result[w, k] = torch.mean(hit.float())
        if k >= 1:
            result[:, k] += result[:, k-1]
    return result.numpy()


def save_accuracy(fpath, accu):
    buffer = "k\window," + ",".join([str(ele+1) for ele in range(accu.shape[0])]) + "\n"
    for k in range(accu.shape[1]):
        buffer += str(k+1) + "," + ",".join([str(ele) for ele in accu[:, k]]) + "\n"
    with open(fpath, "w") as f:
        f.write(buffer)
    return
