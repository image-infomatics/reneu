from reneu.lib.segmentation import RegionGraph 
from reneu.lib.segmentation import watershed 

import h5py
import os
import tifffile
import numpy as np



def segment(affs: np.ndarray, seg: np.ndarray, threshold: float):
    print('construct region graph...')
    rg = RegionGraph(affs, seg)

    print('region graph before segmentation:')
    rg.print() 
    print('gready mean agglomeration...')
    seg=rg.greedy_merge_until(seg, threshold)
    print('region graph after segmentation: ')
    rg.print()

    #print('save results...')
    #tifffile.imwrite(os.path.join(DIR, "seg_rg.tif"), data=seg)
    #with h5py.File(os.path.join(DIR, "seg_rg.h5"), "w") as f:
    #    f['main'] = seg
    print("shape of segmentation: ", seg.shape)
    print(seg)
    
    
# #def test_region_graph():
# #DIR = os.path.join(os.path.dirname(__file__), '../data/')
# #with h5py.File(os.path.join(DIR, "aff_160k.h5"), "r") as f:
# #    affs = np.asarray(f["main"])
# affs = np.zeros((3,10,4,4), dtype=np.float32)
# for z in range(10):
#     affs[0,z,:,:] = z/10
#     affs[1:,z,:,:] = (z+1) / 10


# # print('watershed ...')
# # seg = watershed(affs, 0, 0.9999)
# # print(seg)
# # tifffile.imwrite(os.path.join(DIR, "watershed_basins.tif"), data=seg)
# seg = np.zeros((10,4,4), dtype=np.uint32)
# for z in range(10):
#     seg[z,:,:] = z


def test_region_graph():
    seg = np.arange(4)
    seg = np.reshape(seg, (1,2,2))
    seg += 1

    affs = np.zeros((3,1,2,2), dtype=np.float32)
    affs[0, 0, 0, 1] = 0.9
    affs[1, 0, 1, 0] = 0.6
    affs[0, 0, 1, 1] = 0.8
    affs[1, 0, 1, 1] = 0.5

    print('affinity map: \n', affs)
    segment(affs, seg, 0.7)

