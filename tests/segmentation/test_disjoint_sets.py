from copy import deepcopy
import numpy as np

from reneu.lib.segmentation import DisjointSets, agglomerated_segmentation_to_disjoint_sets


def test_disjoint_sets():
    print('test make and union set...')
    dsets = DisjointSets()
    assert dsets.array.shape[0]==0
    dsets.make_set(1)
    dsets.make_set(2)
    assert dsets.array.shape[0]==0
    dsets.union_set(1, 2)
    root = dsets.find_set(1)
    assert root == 2

    # dsets.make_set(3)
    # dsets.union_set(2, 3)
    # assert dsets.find_set(1) == dsets.find_set(2) == dsets.find_set(3)

    print('test transforming to array...')
    np.testing.assert_array_equal(dsets.array, np.array([[1,2]]))

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
    
    arr2 = dsets.array
    assert arr2.shape == arr.shape
    assert np.all(arr == arr2)

    arr3 = np.arange(7,13, dtype=np.uint64).reshape(3, 2)
    dsets.merge_array(arr3)
    arr4 = np.arange(1,13, dtype=np.uint64).reshape(6, 2)
    np.testing.assert_equal(dsets.array, arr4)

    arr5 = np.array([[1, 1], [2, 2], [1,2], [1, 2]])
    dsets.merge_array(arr5)
    np.testing.assert_equal(arr4, dsets.array)

    assert dsets.find_set(3) == 4
    arr5 = np.array([[1, 3]])
    dsets.merge_array(arr5)
    # arr = dsets.array
    # the original connection should not be broken!
    assert dsets.find_set(3) == 4



def test_agglomerated_segmentation_to_disjoint_sets():
    # frag = np.random.randint(5,5,5)
    frag = np.arange(27, dtype=np.uint64)
    frag = np.reshape(frag, (3,3,3))
    seg = deepcopy(frag)
    seg[1,1,1] = seg[1,1,2]
    seg[2,1,1] = seg[1,1,1]
    # frag = frag.astype(np.uint64)
    # seg = seg.astype(np.uint64)

    dsets = agglomerated_segmentation_to_disjoint_sets(frag, seg)
    breakpoint()
    assert dsets.array.shape[0] == 2 