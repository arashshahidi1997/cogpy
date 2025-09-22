import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def untuple(res):
    if len(res) == 1:
        return res[0]
    else:
        return res
    
def num_to_tuple(tuple_):
    if not isinstance(tuple_, tuple):
        tuple_ = tuple((tuple_,))
        assert len(tuple_) == 1
    return tuple_

def arr_to_num(arr):
    if isinstance(arr, np.ndarray):
        if arr.ndim == 0:
            return arr.item()
        return arr

def init_worker():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def parallelize(func, arg_list):
    # Wrap mtx_per with tqdm for progress reporting
    # Note: tqdm won't display the exact progress for the map operation directly,
    # so we use imap_unordered for approximate progress indication.
    with Pool(initializer=init_worker, processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(func, arg_list), total=len(arg_list)))
    return results

