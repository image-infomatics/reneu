from copy import deepcopy
import numpy as np
from numpy.random import rand

from reneu.lib.segmentation import seeded_watershed, remove_contact, fill_background_with_affinity_guidance


def test_seeded_watershed():
    pass

def test_remove_contact():
    seg1 = np.zeros((2,4,4), dtype=np.uint64)
    seg1[..., :2] = 1
    seg1[..., 2:] = 2

    seg2 = deepcopy(seg1)
    remove_contact(seg2)
    assert np.any(seg1!=seg2)


def test_fill_segmentation_with_affinity_guidance():
    affs = rand(3,4,4,4)
    affs = affs.astype(np.float32)
    seg = np.random.randint(0, high=8, size=affs.shape[1:], dtype=np.uint64)
    seg2 = deepcopy(seg)
    fill_background_with_affinity_guidance(seg2, affs, 0.5)
    assert np.any(seg!=seg2)
