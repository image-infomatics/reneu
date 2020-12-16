from cloudvolume.lib import Bbox
from reneu.lib.binary_bounding_box_tree import BinaryBoundingBoxTree


def test_tree():
    bbox = Bbox.from_list([0, 0, 0, 8, 8, 8])
    bbbt = BinaryBoundingBoxTree(bbox, (8,8,8))
    ot = bbbt.order2tasks 
    assert len(ot) == 1

    bbox = Bbox.from_list([0, 0, 0, 8, 8, 8])
    bbbt = BinaryBoundingBoxTree(bbox, (2,2,2))
    ot = bbbt.order2tasks
    # print(ot)
    assert len(ot) == 7

