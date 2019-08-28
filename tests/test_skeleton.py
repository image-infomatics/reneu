from os.path import join as joinpath
from os.path import dirname

from reneu.skeleton import Skeleton

def test_skeleton():
    sk = Skeleton.from_swc(joinpath(dirname(__file__), '../data/Nov10IR3e.CNG.swc'))
    assert len(sk) == 2277
    print('number of nodes: ', len(sk))