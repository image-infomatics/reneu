
from collections import defaultdict
import numpy as np

from cloudvolume.lib import Bbox, Vec
import cc3d

from .test_region_graph import agglomerate, get_random_affinity_map
from reneu.lib.binary_bounding_box_tree import BinaryBoundingBoxTree
from reneu.lib.segmentation import RegionGraph, watershed, RegionGraphChunk, Dendrogram

from sklearn.metrics import rand_score


def test_region_graph_chunk():
    sz = 8
    affs = get_random_affinity_map(sz)
    fragments = watershed(affs, 0, 0.9)

    # split the chunks, so the contacting surface 
    # do not have continuous segmentation id
    lower_fragments = fragments[:, :, :sz//2]
    lower_fragments = cc3d.connected_components(
        lower_fragments, connectivity=6)
    upper_fragments = fragments[:, :, sz//2:]
    upper_fragments = cc3d.connected_components(
        upper_fragments, connectivity=6)
    upper_fragments[upper_fragments>0] += np.max(lower_fragments)
    fragments[:, :, :sz//2] = lower_fragments
    fragments[:, :, sz//2:] = upper_fragments
    print('fragments: \n', fragments)

    threshold = 0.3
    rg = RegionGraph(affs, fragments)
    print('gready mean agglomeration...')
    dend = rg.greedy_merge(fragments, threshold)
    seg = dend.materialize(fragments, threshold)

    print('\nsegmentation: \n', seg)

    bbox = Bbox.from_list([0, 0, 0, 1, sz, sz])
    bbbt = BinaryBoundingBoxTree(bbox, (1, sz, sz//2))
    order2tasks = bbbt.order2tasks

    rgcs = defaultdict(dict)
    dends = defaultdict(dict)
    for order in range(len(order2tasks)):
        tasks = order2tasks[order]
        # bbox to task
        if order == 0:
            for bbox, task in tasks.items():
                boundary_flags = task[0]
                leaf_affs = affs[:,
                    bbox.minpt[0] : bbox.maxpt[0],
                    bbox.minpt[1] : bbox.maxpt[1],
                    bbox.minpt[2] : bbox.maxpt[2],
                ]
                
                # if the face is not a volume boundary 
                # we'll cutout the contacting face as well
                # it has a contacting chunk face
                offset = Vec(*[f-1 for f in boundary_flags[:3]])
                leaf_fragments = fragments[
                    bbox.minpt[0] + offset[0] : bbox.maxpt[0],
                    bbox.minpt[1] + offset[1] : bbox.maxpt[1],
                    bbox.minpt[2] + offset[2] : bbox.maxpt[2]
                ]
                region_graph_chunk = RegionGraphChunk(leaf_affs, leaf_fragments, boundary_flags)
                dend = region_graph_chunk.merge_in_leaf_chunk(threshold)
                print('dendrogram in leaf node: ', dend)
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
                print('dendrogram from inode: ', dend)
                rgcs[order][bbox] = lower_rgc
                dends[order][bbox] = dend

    combined_dend = Dendrogram();
    for order, bbox2dend in dends.items():
        for bbox, dend in bbox2dend.items():
            combined_dend.merge(dend)
    print('combined dendrogram: ', combined_dend)
    seg2 = combined_dend.materialize(fragments, threshold)
    score = rand_score(seg.flatten(), seg2.flatten())
    print('rand score: ', score)
    print('fragments: \n', fragments)
    print('original segmentation: \n', seg)
    print('distributed segmentation: \n', seg2)
    # assert score == 1
    breakpoint()


# def test_rand_score():
#     sz = 64
#     seg = np.arange(sz*sz*sz, dtype=np.uint64).reshape((sz,sz,sz))

    # print('rand score: ', rand_score(seg.flatten(), seg.flatten()))



        

