import pickle
from reneu.lib.segmentation import RegionGraph 
from reneu.lib.segmentation import watershed, fill_background_with_affinity_guidance 

# import h5py
# import os
# import tifffile
import numpy as np

np.random.seed(0)


def agglomerate(affs: np.ndarray, seg: np.ndarray, affinity_threshold: float = 0., 
        voxel_num_threshold: int=18446744073709551615):
    """
    Parameters:
    voxel_num_threshold [int]: the default value is the maximum limit of C++ size_t
    """
    print('construct region graph...')
    rg = RegionGraph(affs, seg)

    print('region graph as array: \n', rg.array)

    print('region graph before segmentation:', rg)
    print('gready mean agglomeration...')
    dend = rg.greedy_merge(seg, affinity_threshold, voxel_num_threshold)
    seg = dend.materialize(seg, affinity_threshold)
    print('region graph after segmentation: ', rg)

    print("shape of segmentation: ", seg.shape)
    # print(seg)

    print('dendrogram as array: \n', dend.array)
    return seg


def test_agglomeration():
    seg = np.arange(4, dtype=np.uint64)
    seg = np.reshape(seg, (1,2,2))
    seg += 1

    affs = np.zeros((3,1,2,2), dtype=np.float32)
    affs[0, 0, 0, 1] = 0.9
    affs[1, 0, 1, 0] = 0.6
    affs[0, 0, 1, 1] = 0.8
    affs[1, 0, 1, 1] = 0.5

    print('affinity map: \n', affs)
    seg = agglomerate(affs, seg, affinity_threshold = 0.7)

    print('segmentation after agglomeration: ', seg)
    
    np.testing.assert_array_equal(seg, np.array([[[2,2],[4,4]]]))


def random_2d_affinity_map(sz: int):
    # make sure that the random array is consistent
    np.random.seed(23)
    affs = np.random.rand(3, 1, sz, sz).astype(np.float32)
    affs[2,...] = 0
    print('random affinity map \n: ', affs)
    return affs

def random_3d_affinity_map(sz: tuple):
    # make sure that the random array is consistent
    np.random.seed(23)
    affs = np.random.rand(3, *sz).astype(np.float32)
    # print('random affinity map \n: ', affs)
    return affs 

def test_watershed_and_fill_background():
    affs = random_2d_affinity_map(3)
    seg = watershed(affs, 0, 0.9)
    np.testing.assert_array_equal(seg, np.array([[[1,1,1], [2,3,3], [2,3,3]]]))

    ws_seg = np.copy(seg)
    fill_background_with_affinity_guidance(seg, affs)
    print('filled voxel number: ', np.count_nonzero(ws_seg - seg))
   

def test_random_agglomeration():
    sz = 4
    affs = random_2d_affinity_map(sz)
    seg = np.arange(sz*sz, dtype=np.uint64).reshape((1,sz,sz))

    seg = agglomerate(affs, seg, affinity_threshold = 0.3)
    np.testing.assert_array_equal(seg,
            np.array([[[0, 2, 2,  3], 
                       [2, 2, 11, 11],
                       [2, 9, 11, 11],
                       [2, 11,11, 11]]])
    )

def test_merge_small_fragments():
    sz = (32, 32, 32)
    voxel_num_threshold = 4
    affinity_threshold = 0.3

    affs = random_3d_affinity_map(sz)
    # seg = np.arange(np.product(sz), dtype=np.uint64).reshape((sz))
    seg0 = watershed(affs, 0., 0.9999)
    segids0, counts0 = np.unique(seg0[seg0>0], return_counts=True)
    # merge small fragments
    seg1 = agglomerate(affs, seg0, 
        affinity_threshold=affinity_threshold)

    segids1, counts1 = np.unique(seg1[seg1>0], return_counts=True)
    assert len(segids1) < len(segids0)
    # assert np.min(counts1) > voxel_num_threshold

    # rg = RegionGraph(affs, seg)
    # dend = rg.merge_small_fragments(seg, 4)
    # seg2 = dend.materialize(seg, 0.)
    seg2 = agglomerate(affs, seg1, voxel_num_threshold=voxel_num_threshold)
    segids2, counts2 = np.unique(seg2[seg2>0], return_counts=True)
    assert np.min(counts2) > voxel_num_threshold
    assert len(segids2) < len(segids1)
    # the agglomeration should not be too aggressive
    assert len(segids2) > 4

# def test_segment_large_affinity_map():
#     DIR = os.path.join(os.path.dirname(__file__), '../data/')
#     with h5py.File(os.path.join(DIR, "aff_160k.h5"), "r") as f:
#         affs = np.asarray(f["main"])
    
#     print('watershed ...')
#     seg = watershed(affs, 0, 0.9999)
#     tifffile.imwrite(os.path.join(DIR, "watershed_basins.tif"), data=seg)
#     # ws_seg = np.copy(seg)
#     # fill_background_with_affinity_guidance(seg, affs)
#     # print('filled voxel number: ', np.count_nonzero(ws_seg - seg))

#     print('agglomeration...')
#     seg = agglomerate(affs, seg, 0.5)
#     print('save results...')
#     tifffile.imwrite(os.path.join(DIR, "seg_rg.tif"), data=seg)
#     with h5py.File(os.path.join(DIR, "seg_rg.h5"), "w") as f:
#         f['main'] = seg

def test_pickle():
    affs = random_2d_affinity_map(8)
    seg = watershed(affs, 0, 0.9)
    rg = RegionGraph(affs, seg)
    data = pickle.dumps(rg)
    rg2 = pickle.loads(data)
    # although the rg and rg2 is the same, the binary representation
    #  is different due to hash compatibility is not guaranteed
    # assert data == pickle.dumps(rg2)