import numpy as np

from chunkflow.chunk import Chunk

from reneu.lib.segmentation import even_dilation


def test_even_dilation():
    seg = Chunk.create(dtype=np.uint64, pattern='sin')
    breakpoint()
    # seg2 = seg.clone()
    seg2 = even_dilation(seg.array)

    assert np.any(seg.array == 0)
    assert np.any(seg.array!=seg2)
    assert np.all(seg2>0)