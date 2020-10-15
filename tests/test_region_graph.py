from reneu.lib.segmentation import RegionGraph 
from reneu.lib.segmentation import watershed 

import h5py
import os
import tifffile
import numpy as np

#def test_region_graph():
DIR = os.path.join(os.path.dirname(__file__), '../data/')
with h5py.File(os.path.join(DIR, "aff_160k.h5"), "r") as f:
    affs = np.asarray(f["main"])

print('watershed ...')
seg = watershed(affs, 0, 0.9999)
tifffile.imwrite(os.path.join(DIR, "watershed_basins.tif"), data=seg)

print('construct region graph...')
rg = RegionGraph(affs, seg) 
print('gready mean agglomeration...')
seg=rg.greedy_merge_until(seg, 0.5)

print('save results...')
tifffile.imwrite(os.path.join(DIR, "seg_rg.tif"), data=seg)
with h5py.File(os.path.join(DIR, "seg_rg.h5"), "w") as f:
    f['main'] = seg
