#!/usr/bin/env python

from copy import deepcopy
import numpy as np
from numpy.random import rand

from reneu.lib.segmentation import seeded_watershed, remove_contact, fill_background_with_affinity_guidance
from reneu.segmentation import seeded_watershed_2d


def random_affinity_map(shape=(3,4,4,4)):
    if isinstance(shape, int):
        shape = (3, shape, shape, shape)
    elif len(shape)==3:
        shape = tuple(3, *shape)
    else:
        assert len(shape) == 4
        assert shape[0] == 3

    affs = rand(*shape)
    affs = affs.astype(np.float32)
    return affs

def random_segmentation(shape=(4,4,4), high=8):
    if isinstance(shape, int):
        shape = (shape, shape, shape)
    else:
        assert len(shape) == 3

    seg = np.random.randint(0, high=high, size=shape, dtype=np.uint64)
    return seg

def test_seeded_watershed(shape=96):
    affs = random_affinity_map(shape=shape)
    seg = random_segmentation(shape=shape)

    seg2 = deepcopy(seg)
    seeded_watershed_2d(seg2, affs, 0.5)
    assert np.any(seg!=seg2)


def test_remove_contact():
    seg1 = np.zeros((2,4,4), dtype=np.uint64)
    seg1[..., :2] = 1
    seg1[..., 2:] = 2

    seg2 = deepcopy(seg1)
    remove_contact(seg2)
    assert np.any(seg1!=seg2)


def test_fill_segmentation_with_affinity_guidance():
    affs = random_affinity_map()
    seg = random_segmentation()
    seg2 = deepcopy(seg)
    fill_background_with_affinity_guidance(seg2, affs, 0.5)
    assert np.any(seg!=seg2)


if __name__ == '__main__':
    test_seeded_watershed()