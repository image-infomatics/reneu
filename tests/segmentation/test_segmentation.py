import numpy as np

from chunkflow.chunk import Chunk

from reneu.segmentation import contacting_and_inner_obj_ids


def test_contacting_and_inner_obj_ids():
    seg = Chunk.create(size=(19, 19, 19), pattern='random', dtype=np.float32)
    seg = seg.connected_component(0.5, 6)
    contacting, inner = contacting_and_inner_obj_ids(seg)

    assert len(contacting)>0
    assert len(inner)>0