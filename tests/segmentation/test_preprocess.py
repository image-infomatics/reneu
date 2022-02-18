from copy import deepcopy
import numpy as np
from reneu.segmentation import remove_contact

def test_remove_contact():
    seg1 = np.zeros((4,4,4), dtype=np.uint64)
    seg1[:2, ...] = 1
    seg1[2:, ...] = 2

    seg2 = deepcopy(seg1)
    remove_contact(seg2)
    assert np.any(seg1!=seg2)