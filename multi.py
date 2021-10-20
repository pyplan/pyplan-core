import time
import multiprocessing 
import numpy as np

def getNode(nodeId):
    size = [200,100,1000]
    a = np.random.randint(1,10,size) * np.random.randint(1,10,size)
    del a


def simple(inputs):
    starttime = time.time()
    [getNode(xx) for xx in inputs]
    endtime = time.time()
    print(f'Simple core {endtime - starttime} seconds')


def multi_core(inputs):
    starttime = time.time()
    pool = multiprocessing.Pool(processes=8)
    pool.map(getNode, inputs)
    pool.close()
    endtime = time.time()
    print(f'Multi core {endtime - starttime} seconds')

def multi_process(inputs):
    starttime = time.time()
    for input in inputs:
        p = multiprocessing.Process(target=getNode, args=(input,))
        p.start()
        p.join()
    endtime = time.time()
    print(f'Multi process {endtime - starttime} seconds')

nodes = 20

#simple([f'node{xx}' for xx in range(nodes)])

multi_core([f'node{xx}' for xx in range(nodes)])

#multi_process([f'node{xx}' for xx in range(nodes)])
