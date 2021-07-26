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

    # dsets.make_set(3)
    # dsets.union_set(2, 3)
    # assert dsets.find_set(1) == dsets.find_set(2) == dsets.find_set(3)

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
    
    arr2 = dsets.to_array()
    assert arr2.shape == arr.shape
    assert np.all(arr == arr2)

    arr3 = np.arange(7,13, dtype=np.uint64).reshape(3, 2)
    dsets.merge_array(arr3)
    arr4 = np.arange(1,13, dtype=np.uint64).reshape(6, 2)
    np.testing.assert_equal(dsets.to_array(), arr4)

    arr5 = np.array([[1, 1], [2, 2], [1,2], [1, 2]])
    dsets.merge_array(arr5)
    np.testing.assert_equal(arr4, dsets.to_array())

    assert dsets.find_set(3) == 4
    arr5 = np.array([[1, 3]])
    dsets.merge_array(arr5)
    # arr = dsets.to_array()
    # the original connection should not be broken!
    assert dsets.find_set(3) == 4

