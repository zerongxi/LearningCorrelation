import multiprocessing as mp
import numpy as np
import ctypes
import itertools


def worker(func, data_tuples, id, ret_dict):
    data = [get_array_from_shared_memory(*ele) for ele in data_tuples]
    ret_dict[id] = func(*data)
    return


def pool_by_shared_array(func, n_processes, split_data=None, share_data=None):

    shared = [put_array_into_shared_memory(ele, False) for ele in share_data]

    semaphore = mp.Semaphore(n_processes)
    manager = mp.Manager()
    ret_dict = manager.dict()
    processes = []
    for cnt in range(len(split_data)):
        split = [put_array_into_shared_memory(ele, False) for ele in split_data]
        merged = split + shared
        semaphore.acquire()
        p = mp.Process(
            target=worker,
            args=(func, merged, cnt, ret_dict,)
        )
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    for cnt in range(len(ret_dict)):
        if ret_dict[cnt] is None:
            return None
    
    if type(ret_dict[0]) == tuple:
        ret = []
        for cnt in range(len(ret_dict[0])):
            ret.append(np.concatenate([ret_dict[i][cnt] for i in range(len(ret_dict))], 0))
    else:
        ret = np.concatenate([ret_dict[i] for i in range(len(ret_dict))], 0)
    return ret


def put_array_into_shared_memory(arr, lock):
    ctypes_ = dict(float32=ctypes.c_float, int64=ctypes.c_int64)
    dtypes_ = dict(float32=np.float32, int64=np.int64)

    wrapper = mp.Array if lock else mp.RawArray
    memory = wrapper(ctypes_[arr.dtype.name], arr.size)
    shape = arr.shape
    dtype = dtypes_[arr.dtype.name]
    return memory, dtype, shape


def get_array_from_shared_memory(memory, dtype, shape):
    arr = np.frombuffer(memory, dtype).reshape(shape)
    return arr


def split_arr_evenly(arr, split, axis=0):
    if axis != 0:
        arr = np.swapaxes(arr, 0, axis)
    ret = []
    interval = int(arr.shape[0] / split)
    for i in range(split):
        ret.append(arr[i * interval:(i + 1) * interval])
        ret[-1] = np.swapaxes(ret[-1], 0, axis)

    return ret


def product_paras(paras):
    keys = list(paras.keys())
    for key in keys:
        if type(paras[key]) is not list:
            paras[key] = [paras[key]]
    ret = list(itertools.product(*[paras[key] for key in keys]))
    ret = [tuple((keys[i], ret[j][i]) for i in range(len(keys))) for j in range(len(ret))]
    ret = [dict(ele) for ele in ret]
    return ret


def dictmap(func, args):
    return func(**args)


def pool_experiments(experiment, paras, n_processes=None):
    if n_processes is None:
        n_processes = max(mp.cpu_count() - 2, 2)
    paras = product_paras(paras)
    print("%d experiments running in %d cores!" % (len(paras), n_processes))
    pool = mp.Pool(processes=n_processes)
    tasks = pool.starmap_async(dictmap, [(experiment, ele) for ele in paras])
    tasks.wait()
    return
