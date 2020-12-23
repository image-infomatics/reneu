
from collections import defaultdict
import numpy as np

from cloudvolume.lib import Bbox

from .test_region_graph import agglomerate, get_random_affinity_map
from reneu.lib.binary_bounding_box_tree import BinaryBoundingBoxTree
from reneu.lib.segmentation import RegionGraph, watershed, RegionGraphChunk, Dendrogram

from sklearn.metrics import rand_score


def test_region_graph_chunk():
    sz = 32
    affs = get_random_affinity_map(sz)
    fragments = watershed(affs, 0, 0.9)

    threshold = 0.3
    rg = RegionGraph(affs, fragments)
    print('gready mean agglomeration...')
    dend = rg.greedy_merge(fragments, threshold)
    seg = dend.materialize(fragments, threshold)

    print('\nsegmentation: \n', seg)

    bbox = Bbox.from_list([0, 0, 0, sz, sz, sz])
    bbbt = BinaryBoundingBoxTree(bbox, (16, 32, 32))
    order2tasks = bbbt.order2tasks

    rgcs = defaultdict(dict)
    dends = defaultdict(dict)
    for order in range(len(order2tasks)):
        tasks = order2tasks[order]
        # bbox to task
        if order == 0:
            for bbox, task in tasks.items():
                boundary_flags = task[0]
                region_graph_chunk = RegionGraphChunk(affs, fragments, boundary_flags)
                dend = region_graph_chunk.merge_in_leaf_chunk(threshold)
                rgcs[order][bbox] = region_graph_chunk
                dends[order][bbox] = dend
        else:
            for bbox, task in tasks.items():
                boundary_flags = task[0]
                split_dim = task[1]
                lower_bbox = task[2]
                upper_bbox = task[3]
                lower_rgc = rgcs[order-1][lower_bbox]
                upper_rgc = rgcs[order-1][upper_bbox]
                dend = lower_rgc.merge_upper_chunk(upper_rgc, split_dim, threshold)
                rgcs[order][bbox] = lower_rgc
                dends[order][bbox] = dend

    combinedDend = Dendrogram();
    for order, bbox2dend in dends.items():
        for bbox, dend in bbox2dend.items():
            combinedDend.merge(dend)
    
    seg2 = dend.materialize(fragments, threshold)
    score = rand_score(seg.flatten(), seg2.flatten())
    print('rand score: ', score)





        

