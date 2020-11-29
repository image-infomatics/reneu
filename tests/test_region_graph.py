from reneu.lib.segmentation import RegionGraph 
from reneu.lib.segmentation import watershed, fill_background_with_affinity_guidance 

import h5py
import os
import tifffile
import numpy as np

np.random.seed(0)


def agglomerate(affs: np.ndarray, seg: np.ndarray, threshold: float):
    print('construct region graph...')
    rg = RegionGraph(affs, seg)

    print('region graph before segmentation:')
    # rg.print() 
    print('gready mean agglomeration...')
    dend = rg.greedy_merge_until(seg, threshold)
    seg = dend.materialize(seg, threshold)
    print('region graph after segmentation: ')
    # rg.print()

    print("shape of segmentation: ", seg.shape)
    # print(seg)

    print('dendrogram as array: \n', dend.array)
    return seg


def test_agglomeration():
    seg = np.arange(4)
    seg = np.reshape(seg, (1,2,2))
    seg += 1

    affs = np.zeros((3,1,2,2), dtype=np.float32)
    affs[0, 0, 0, 1] = 0.9
    affs[1, 0, 1, 0] = 0.6
    affs[0, 0, 1, 1] = 0.8
    affs[1, 0, 1, 1] = 0.5

    print('affinity map: \n', affs)
    seg = agglomerate(affs, seg, 0.7)

    print('segmentation after agglomeration: ', seg)
    
    np.testing.assert_array_equal(seg, np.array([[[2,2],[4,4]]]))


def get_random_affinity_map(sz: int):
    affs = np.random.rand(3,1,sz,sz).astype(np.float32)
    affs[2,...] = 0
    print('random affinity map \n: ', affs)
    return affs


def test_watershed_and_fill_background():
    affs = get_random_affinity_map(3)
    seg = watershed(affs, 0, 0.9)
    np.testing.assert_array_equal(seg, np.array([[[1,1,1], [1,1,2], [2,2,2]]]))

    ws_seg = np.copy(seg)
    fill_background_with_affinity_guidance(seg, affs)
    print('filled voxel number: ', np.count_nonzero(ws_seg - seg))
   

def test_random_agglomeration():
    sz = 4
    affs = get_random_affinity_map(sz)
    seg = np.arange(sz*sz, dtype=np.uint32).reshape((1,sz,sz))

    seg = agglomerate(affs, seg, 0.3)
    np.testing.assert_array_equal(seg,
            np.array([[[0,  11, 11,  7], 
                       [11, 11, 11,  7],
                       [11, 11, 11, 11],
                       [12, 11, 11, 11]]])
    )

#def test_segment_large_affinity_map():
#    DIR = os.path.join(os.path.dirname(__file__), '../data/')
#    with h5py.File(os.path.join(DIR, "aff_160k.h5"), "r") as f:
#        affs = np.asarray(f["main"])
#    
#    print('watershed ...')
#    seg = watershed(affs, 0, 0.9999)
#    tifffile.imwrite(os.path.join(DIR, "watershed_basins.tif"), data=seg)
#    # ws_seg = np.copy(seg)
#    # fill_background_with_affinity_guidance(seg, affs)
#    # print('filled voxel number: ', np.count_nonzero(ws_seg - seg))
#
#    print('agglomeration...')
#    seg = agglomerate(affs, seg, 0.5)
#    print('save results...')
#    tifffile.imwrite(os.path.join(DIR, "seg_rg.tif"), data=seg)
#    with h5py.File(os.path.join(DIR, "seg_rg.h5"), "w") as f:
#        f['main'] = seg