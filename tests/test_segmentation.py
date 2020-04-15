from reneu.xiuli import XSupervoxelDendrogram as SupervoxelDendrogram
import h5py
import os
import tifffile
import numpy as np

def test_supervoxel_dendrogram():
    DIR = os.path.expanduser("~/seungmount/research/Jingpeng/02_pytorch/datasets/CREMI/seung/")
    #DIR = os.path.expanduser("/tmp/")
    with h5py.File(os.path.join(DIR, "affs.h5")) as f:
        affs = np.asarray(f["main"])
    with h5py.File(os.path.join(DIR, "fragments.h5")) as f:
        fragments = np.asarray(f["main"])
        
    dend = SupervoxelDendrogram(affs, fragments, 0.3)
    
    seg = dend.segment(0.3)

    tifffile.imwrite(os.path.join(DIR, "seg.tif"), data=seg)
    
    with h5py.File(os.path.join(DIR, "seg.h5"), "w") as f:
        f.create_dataset("main", seg)


