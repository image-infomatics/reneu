import numpy as np

from chunkflow.chunk import Chunk
from chunkflow.lib.bounding_boxes import Cartesian


from reneu.segmentation import contacting_and_inner_obj_ids, get_nonzero_bounding_box

from reneu.lib.segmentation import get_label_map, DisjointSets

def test_get_nonzero_bounding_box():
    x = np.arange(125, dtype=np.uint64)
    x = x.reshape(5,5,5)
    bbox = get_nonzero_bounding_box(x)
    assert bbox.start == Cartesian(0, 0, 0)
    assert bbox.stop == Cartesian(5,5,5)

    x = np.zeros((5,5,5), dtype=np.uint64)
    tmp = np.arange(27, dtype=np.uint64)
    tmp = tmp.reshape(3,3,3)
    x[1:4, 1:4, 1:4] = tmp
    bbox = get_nonzero_bounding_box(x)
    assert bbox.start == Cartesian(1,1,1)
    assert bbox.stop == Cartesian(4,4,4)

def test_contacting_and_inner_obj_ids():
    seg = Chunk.create(size=(19, 19, 19), pattern='random', dtype=np.float32)
    seg = seg.connected_component(0.5, 6)
    contacting, inner = contacting_and_inner_obj_ids(seg)

    assert len(contacting)>0
    assert len(inner)>0

def test_get_label_map():
    frag = np.arange(27, dtype=np.uint64)
    frag = frag.reshape((3,3,3))

    seg = np.copy(frag)
    seg[1,1,1] = seg[1,0,1]
    seg[0,1,1] = seg[0,0,1]
    label_map = get_label_map(frag, seg)

    expected_label_map = np.array(
        [[ 4,  1],
        [13, 10]], dtype=np.uint64)

    np.testing.assert_array_equal(label_map, expected_label_map)

    dsets = DisjointSets()
    # since dsets always merge the right one to the left one, 
    # we need to switch the left and right to make it consisitent
    expected_label_map[:, [0,1]] = expected_label_map[:, [1,0]]
    dsets.merge_array(expected_label_map)
    seg2 = dsets.relabel(frag)
    np.testing.assert_array_equal(seg, seg2)