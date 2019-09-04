from os.path import join as joinpath
from os.path import dirname
import time
#import numpy as np
from math import isclose

#from reneu.skeleton import Skeleton
from reneu.lib.xiuli import Skeleton

def test_skeleton():
    NEURON_NAME = 'Nov10IR3e.CNG'
    #NEURON_NAME = '77337930247110714'
    start = time.process_time()
    sk = Skeleton(joinpath(dirname(__file__), '../data/{}.swc'.format(NEURON_NAME)))
    print('read time elapse: ', time.process_time() - start)

    start = time.process_time()
    path_length = sk.path_length
    assert isclose( path_length,  3516.6, abs_tol=0.1)
    print('path length: ', path_length)
    print('time elapse of computing path length: ', time.process_time()-start)

    node_num1 = len(sk)

    start = time.process_time()
    #sk.downsample(1000.0)
    sk.downsample(2.0)
    print('time elapsed of downsample: ', time.process_time()-start)
    node_num2 = len(sk)
    print('downsampled from {} nodes to {} nodes.'.format(node_num1, node_num2))

    start = time.process_time()
    sk.write_swc('/tmp/{}.swc'.format(NEURON_NAME), 3)
    print('time elapse of write swc: ', time.process_time() - start)