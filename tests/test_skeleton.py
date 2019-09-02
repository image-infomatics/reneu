from os.path import join as joinpath
from os.path import dirname

#from reneu.skeleton import Skeleton
from reneu.lib.xiuli import Skeleton

def test_skeleton():
    NEURON_NAME = 'Nov10IR3e.CNG'
    sk = Skeleton(joinpath(dirname(__file__), '../data/{}.swc'.format(NEURON_NAME)))
    assert len(sk) == 2277
    print('number of nodes: ', len(sk))

    #sk.downsample(step=1)
    sk.write_swc('/tmp/{}.swc'.format(NEURON_NAME), 3)