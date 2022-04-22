import numpy as np
from reneu.lib.segmentation import fragment_id_map


def test_fragment_id_map():
    # frag = np.random.randint(5,5,5)
    frag = np.arange(125, dtype=np.uint64)
    seg = np.arange(125, dtype=np.uint64)
    frag = np.reshape(frag, (5,5,5))
    seg = np.reshape(seg, (5,5,5))
    # frag = frag.astype(np.uint64)
    # seg = seg.astype(np.uint64)

    id_map = fragment_id_map(frag, seg)
    assert id_map[1] == 1
    # the background voxel 0 should not be mapped
    assert len(id_map) == 124