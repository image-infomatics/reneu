from copy import deepcopy
import pickle

import numpy as np

from reneu.lib.segmentation import DisjointSets, agglomerated_segmentation_to_merge_pairs


def test_disjoint_set_root():
    dsets = DisjointSets()
    N = 9
    for idx in range(1, N):
        dsets.make_set(idx)
    for idx in range(1, N-1):
        dsets.union_set(idx, idx+1)
    
    for idx in range(1, N):
        assert dsets.find_set(idx) == 1

    for idx in range(N+1, 3*N):
        dsets.make_and_union_set(idx, idx+1)

    assert dsets.find_set(N+1) == N+1

    dsets.union_set(N, N+4)
    assert dsets.find_set(N+4) == N+1

    # test pickle
    data = pickle.dumps(dsets)
    dsets2 = pickle.loads(data)
    np.testing.assert_array_equal(dsets.array, dsets2.array)

def test_disjoint_sets():
    print('test make and union set...')
    dsets = DisjointSets()
    assert dsets.array.shape[0]==0
    dsets.make_set(1)
    dsets.make_set(2)
    assert dsets.array.shape[0]==2
    dsets.union_set(1, 2)
    assert dsets.find_set(1) == 1
    assert dsets.find_set(2) == 1

    
    # this is not working yet
    print('test serialization using pickle')
    data = pickle.dumps(dsets)
    dsets2 = pickle.loads(data)
    np.testing.assert_array_equal(dsets.array, dsets2.array)

    # dsets.make_set(3)
    # dsets.union_set(2, 3)
    # assert dsets.find_set(1) == dsets.find_set(2) == dsets.find_set(3)

    print('test transforming to array...')
    np.testing.assert_array_equal(
        dsets.array, 
        np.array([[1,1],[2,1]]))

    print('test relabel...')
    seg = np.arange(0, 27).reshape(3,3,3)
    seg2 = np.copy(seg)
    seg2 = dsets.relabel(seg2)
    seg3 = deepcopy(seg)
    seg3[0, 0, 2] = 1
    np.testing.assert_array_equal(seg2, seg3)

def test_disjoint_sets_merge_array():
    print('test merge seg pairs in an array...')
    arr = np.arange(1,7, dtype=np.uint64).reshape(3, 2)
    dsets = DisjointSets()
    dsets.merge_array(arr, has_root=False)
    assert dsets.find_set(2) == 1
    assert dsets.find_set(4) == 3
    assert dsets.find_set(6) == 5
    
    arr2 = dsets.array
    assert arr2.shape[0] == np.prod(arr.shape)

    arr3 = np.arange(7,13, dtype=np.uint64).reshape(3, 2)
    dsets.merge_array(arr3, has_root=False)
    arr4 = np.arange(1,13, dtype=np.uint64).reshape(6, 2)
    # np.testing.assert_equal(dsets.array, arr4)

    arr5 = np.array([[1, 1], [2, 2], [1,2], [1, 2]])
    dsets.merge_array(arr5)

    assert dsets.find_set(4) == 3
    arr5 = np.array([[1, 3]])
    dsets.merge_array(arr5)
    # arr = dsets.array
    # the original connection should not be broken!
    assert dsets.find_set(4) == 1

def test_agglomerated_segmentation_to_merge_pair():
    # frag = np.random.randint(5,5,5)
    frag = np.arange(27, dtype=np.uint64)
    frag = np.reshape(frag, (3,3,3))
    seg = deepcopy(frag)
    seg[1,1,1] = seg[1,1,2]
    seg[2,2,1] = seg[1,2,1]

    merge_pairs = agglomerated_segmentation_to_merge_pairs(frag, seg)
    # print(f'merge pairs: {merge_pairs}')
    # breakpoint()
    assert merge_pairs.shape[0] == 26 

def test_frag_seg_to_disjoint_sets():
    # frag = np.random.randint(5,5,5)
    frag = np.arange(27, dtype=np.uint64)
    frag = np.reshape(frag, (3,3,3))
    seg = deepcopy(frag)
    seg[1,1,1] = seg[1,1,2]
    seg[2,2,1] = seg[1,2,1]

    dsets = DisjointSets(frag, seg)
    # print(dsets.array)
    arr = dsets.array
    assert arr[13, 1] == 13
    assert arr[24, 1] == 16