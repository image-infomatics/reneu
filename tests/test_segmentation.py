from reneu.lib.segmentation import XSupervoxelDendrogram as SupervoxelDendrogram
from reneu.lib.segmentation import watershed

import h5py
import os
import tifffile
import numpy as np

# def test_supervoxel_dendrogram():
DIR = os.path.join(os.path.dirname(__file__), '../data/')
with h5py.File(os.path.join(DIR, "aff_160k.h5"), "r") as f:
    affs = np.asarray(f["main"])

print('watershed ...')
seg = watershed(affs, 0.5, 0.99)
tifffile.imwrite(os.path.join(DIR, "watershed_basins.tif"), data=seg)

print('read fragments from watershed...')
with h5py.File(os.path.join(DIR, "ffn_seg.h5"), "r") as f:
    fragments = np.asarray(f["seg"])
 
print('construct dendrogram and merge supervoxels')
dend = SupervoxelDendrogram(affs, fragments, 0.7)
seg = dend.segment(0.7)

print('save results...')
tifffile.imwrite(os.path.join(DIR, "seg_test.tif"), data=seg)
with h5py.File(os.path.join(DIR, "seg_test.h5"), "w") as f:
    f['main'] = seg
