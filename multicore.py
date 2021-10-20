import time
import multiprocessing
import numpy as np

from multiprocessing.pool import ThreadPool


def calc(a, b):
    result = (a*b/(b.sum()+1/(a+b))).sum()
    print(result)
    return result




if __name__ == '__main__':
    size = [100, 200, 1000]
    a = np.random.random(size) * np.random.random(size)
    b = np.random.random(size) * np.random.random(size)
    print("Data ready", flush=True)


    print("Single core", flush=True)
    starttime = time.time()
    [calc(a,b) for nn in range(10)]
    endtime = time.time()
    print(f'Result in {endtime - starttime} seconds')

    print("-----------------------------")
    print("Multi core", flush=True)
    starttime = time.time()
    pool = multiprocessing.Pool(processes=6)
    #pool = ThreadPool()
    multiple_results = [pool.apply_async(calc, (a,b)) for xx in range(10)]
    [res.get() for res in multiple_results]
    pool.close()
    endtime = time.time()
    print(f'Multicore in {endtime - starttime} seconds')



