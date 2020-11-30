from reneu.lib.segmentation import fill_background_with_affinity_guidance

import numpy as np 
np.random.seed(0)


def test_fill_background_with_affinity_guidance():
    seg = np.random.randint(2000, dtype=np.uint32, size=(64,64,64))
    seg[seg<500] = 0
    assert np.any(seg==0)
    affs = np.random.rand(3,64,64,64).astype(np.float32)
    fill_background_with_affinity_guidance(seg, affs)
    assert np.alltrue(seg>0)
    
    seg = np.random.randint(2000, dtype=np.uint64, size=(64,64,64))
    seg[seg<500] = 0
    assert np.any(seg==0)
    affs = np.random.rand(3,64,64,64).astype(np.float32)
    fill_background_with_affinity_guidance(seg, affs)
    assert np.alltrue(seg>0)
