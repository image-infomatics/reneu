from os.path import join as joinpath
from os.path import dirname
import time

#from reneu.skeleton import Skeleton
from reneu.lib.xiuli import Skeleton

def test_skeleton():
    NEURON_NAME = 'Nov10IR3e.CNG'
    #NEURON_NAME = '77337930247110714'
    start = time.process_time()
    sk = Skeleton(joinpath(dirname(__file__), '../data/{}.swc'.format(NEURON_NAME)))
    print('read time elapse: ', time.process_time() - start)

    print('number of nodes: ', len(sk))

    start = time.process_time()
    #sk.downsample(1000.0)
    sk.downsample(2.0)
    print('time elapsed of downsample: ', time.process_time()-start)

    start = time.process_time()
    sk.write_swc('/tmp/{}.swc'.format(NEURON_NAME), 3)
    print('time elapse of write swc: ', time.process_time() - start)