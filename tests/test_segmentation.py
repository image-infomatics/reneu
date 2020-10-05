from reneu.lib.segmentation import XSupervoxelDendrogram as SupervoxelDendrogram
from reneu.lib.segmentation import watershed

import h5py
import os
import tifffile
import numpy as np

def test_supervoxel_dendrogram():
    DIR = os.path.join(os.path.dirname(__file__), '../data/')
    with h5py.File(os.path.join(DIR, "aff_160k.h5"), "r") as f:
        affs = np.asarray(f["main"])
    
    print('watershed ...')
    fragments = watershed(affs, 0.6, 0.9999)
    tifffile.imwrite(os.path.join(DIR, "watershed_basins.tif"), data=fragments)

    print('dilate the fragments...')
    #fill_background_with_affinity_guidance(fragments, affs)
    #assert fragments.min() > 0 
    tifffile.imwrite(os.path.join(DIR, "watershed_basins_filled.tif"), data=fragments)

    print('construct dendrogram and merge supervoxels')
    threshold = 0.999
    min_threshold = 0.0
    size_threshold = 32 
    dend = SupervoxelDendrogram(affs, fragments, min_threshold)
    seg = dend.segment(threshold, size_threshold)

    print('save results...')
    tifffile.imwrite(os.path.join(DIR, "seg_test.tif"), data=seg)
    with h5py.File(os.path.join(DIR, "seg_test.h5"), "w") as f:
        f['main'] = seg
    