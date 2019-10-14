from os.path import join as joinpath
from os.path import dirname
import time
from math import isclose


from reneu.lib.libxiuli import XSkeleton
from reneu.skeleton import Skeleton

NEURON_NAME = 'Nov10IR3e.CNG'
#NEURON_NAME = '77337930247110714'
file_name = joinpath(dirname(__file__), '../data/{}.swc'.format(NEURON_NAME))

def test_xskeleton():
    print('\ntest XSkeleton class from c++...')
    start = time.process_time()
    sk = XSkeleton(file_name)
    print('read time elapse: ', time.process_time() - start)

    start = time.process_time()
    path_length = sk.path_length
    if NEURON_NAME == 'Nov10IR3e.CNG':
        # the compared number is from treestoolbox
        assert isclose( path_length,  3516.6, abs_tol=0.1)
    elif NEURON_NAME == '77337930247110714':
        # this number is from first time run, not based on other tools!
        assert isclose( path_length, 3173894.5, abs_tol=0.1 )

    print('path length: ', path_length)
    print('time elapse of computing path length: ', time.process_time()-start)

    node_num1 = len(sk)

    start = time.process_time()
    if NEURON_NAME == 'Nov10IR3e.CNG':
        sk.downsample(2.0)
    elif NEURON_NAME == '77337930247110714':
        sk.downsample(1000.0)
        assert len(sk) == 1476

    print('time elapsed of downsample: ', time.process_time()-start)
    node_num2 = len(sk)
    print('downsampled from {} nodes to {} nodes.'.format(node_num1, node_num2))


def test_skeleton():
    print('\ntest inherited skeleton class in python...')
    start = time.process_time()
    sk = Skeleton.from_swc( file_name )
    print('time elapse in python read_swc: ', time.process_time()-start)

    node_num1 = len(sk)
    if NEURON_NAME == 'Nov10IR3e.CNG':
        sk.downsample(2.0)
    elif NEURON_NAME == '77337930247110714':
        sk.downsample(1000.0)
        assert len(sk) == 1476
    
    node_num2 = len(sk)
    print('downsampled from {} nodes to {} nodes.'.format(node_num1, node_num2))
     
    start = time.process_time()
    temp_file_name = '/tmp/{}.swc'.format(NEURON_NAME)
    sk.to_swc(temp_file_name, 3)

    with open(temp_file_name) as f:
        assert f.read() == sk.to_swc_str(3)

    print('time elapse of write swc: ', time.process_time() - start)
    sk2 = Skeleton.from_swc( temp_file_name )
    assert sk == sk2

    skelbuf = sk.to_precomputed()
    sk3 = Skeleton.from_precomputed( skelbuf );
    assert sk == sk3
   
    print('path length: ', sk.path_length)

    print('number of nodes: ', len(sk))
    assert len(sk) == sk.nodes.shape[0]
