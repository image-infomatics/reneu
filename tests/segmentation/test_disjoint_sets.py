from copy import deepcopy
import numpy as np

from reneu.lib.segmentation import DisjointSets


def test_disjoint_sets():
    print('test make and union set...')
    dsets = DisjointSets()
    dsets.make_set(1)
    dsets.make_set(2)
    dsets.union_set(1, 2)
    root = dsets.find_set(1)
    assert root == 2

    print('test to_array...')
    arr = dsets.to_array()
    np.testing.assert_array_equal(arr, np.array([[1,2]]))

    print('test relabel...')
    seg = np.arange(0, 27).reshape(3,3,3)
    seg2 = dsets.relabel(seg)
    seg3 = deepcopy(seg)
    seg3[0, 0, 1] = 2
    np.testing.assert_array_equal(seg2, seg3)

    print('test merge seg pairs in an array...')
    arr = np.arange(1,7, dtype=np.uint64).reshape(3, 2)
    dsets = DisjointSets()
    dsets.merge_array(arr)
    assert dsets.find_set(1) == 2
    assert dsets.find_set(3) == 4
    assert dsets.find_set(5) == 6

    

