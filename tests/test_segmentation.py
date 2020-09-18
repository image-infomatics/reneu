from reneu.lib.segmentation import XSupervoxelDendrogram as SupervoxelDendrogram
import h5py
import os
import tifffile
import numpy as np

def test_supervoxel_dendrogram():
    DIR = os.path.join(os.path.dirname(__file__), '../data/')
    with h5py.File(os.path.join(DIR, "affs_test.h5"), "r") as f:
        affs = np.asarray(f["main"])
    
    print('read fragments from watershed...')
    with h5py.File(os.path.join(DIR, "fragments_test.h5"), "r") as f:
        fragments = np.asarray(f["main"][:, :300, :300])
     
    print('construct dendrogram and merge supervoxels')
    dend = SupervoxelDendrogram(affs, fragments, 0.3)
    seg = dend.segment(0.3)
    
    print('save results...')
    tifffile.imwrite(os.path.join(DIR, "seg_test.tif"), data=seg)
    with h5py.File(os.path.join(DIR, "seg_test.h5"), "w") as f:
        f['main'] = seg
