from os.path import join as joinpath
from os.path import dirname
import time

#from reneu.skeleton import Skeleton
from reneu.lib.xiuli import Skeleton

def test_skeleton():
    NEURON_NAME = 'Nov10IR3e.CNG'
    start = time.process_time()
    sk = Skeleton(joinpath(dirname(__file__), '../data/{}.swc'.format(NEURON_NAME)))
    print('read time elapse: ', time.process_time() - start)

    assert len(sk) == 2277
    print('number of nodes: ', len(sk))

    sk.downsample(1.0)
    start = time.process_time()
    sk.write_swc('/tmp/{}.swc'.format(NEURON_NAME), 3)
    print('time elapse of write swc: ', time.process_time() - start)