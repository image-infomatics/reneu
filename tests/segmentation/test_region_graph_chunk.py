import os
from tempfile import mkdtemp
import pickle
from shutil import rmtree
import numpy as np
np.random.seed(3817)

from cloudvolume.lib import Bbox, Vec
import cc3d

from reneu.lib.binary_bounding_box_tree import BinaryBoundingBoxTree
from reneu.lib.segmentation import RegionGraph, watershed, RegionGraphChunk, Dendrogram

from sklearn.metrics import rand_score



def get_random_affinity_map(sz: tuple, seed: int=1):
    np.random.seed(seed)
    affs = np.random.rand(3,*sz).astype(np.float32)
    print('random affinity map \n: ', affs)
    return affs

def distributed_agglomeration(fragments: np.ndarray, affs: np.ndarray, threshold: float, chunk_size: tuple):
    rgc_dir = mkdtemp()
    dend_dir = mkdtemp()

    print('\ndistributed agglomeration...')
    bbox = Bbox.from_list([0, 0, 0, *fragments.shape])
    bbbt = BinaryBoundingBoxTree(bbox, chunk_size)
    order2tasks = bbbt.order2tasks

    # bbox2rgc = dict()
    # bbox2dend = dict()
    for order in range(len(order2tasks)):
        tasks = order2tasks[order]
        if order == 0:
            for bbox, task in tasks.items():
                boundary_flags, _, lower_bbox, upper_bbox = task
                assert lower_bbox is None
                assert upper_bbox is None
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
                # breakpoint()
                region_graph_chunk = RegionGraphChunk(leaf_affs, leaf_fragments, boundary_flags)
                # print('region graph in leaf chunk before merging: ', region_graph_chunk)
                dend = region_graph_chunk.merge_in_leaf_chunk(threshold)
                # print('region graph in leaf chunk after merging: ', region_graph_chunk)
                
                # bbox2rgc[bbox] = region_graph_chunk
                # bbox2dend[bbox] = dend
                
                with open(os.path.join(rgc_dir, f'{bbox.to_filename()}.pickle'), 'wb') as f:
                    pickle.dump(region_graph_chunk, f)
                with open(os.path.join(rgc_dir, f'{bbox.to_filename()}.pickle'), 'rb') as f:
                    region_graph_chunk2 = pickle.load(f)
                assert pickle.dumps(region_graph_chunk) == pickle.dumps(region_graph_chunk2)
                # print(region_graph_chunk2)

                # print('dendrogram in leaf node: ', dend)
                with open(os.path.join(dend_dir, f'{bbox.to_filename()}.pickle'), 'wb') as f:
                    pickle.dump(dend, f)
                with open(os.path.join(dend_dir, f'{bbox.to_filename()}.pickle'), 'rb') as f:
                    dend2 = pickle.load(f)
                assert pickle.dumps(dend) == pickle.dumps(dend2)
                # print(dend2)
        else:
            for bbox, task in tasks.items():
                boundary_flags, split_dim, lower_bbox, upper_bbox = task
                
                # lower_rgc = bbox2rgc[lower_bbox]
                # upper_rgc = bbox2rgc[upper_bbox]
                
                with open(os.path.join(rgc_dir, f'{lower_bbox.to_filename()}.pickle'), 'rb') as f:
                    lower_rgc = pickle.load(f)
                with open(os.path.join(rgc_dir, f'{upper_bbox.to_filename()}.pickle'), 'rb') as f:
                    upper_rgc = pickle.load(f)

                # print('lower region graph chunk: ', lower_rgc)
                # print('upper region graph chunk: ', upper_rgc)
                dend = lower_rgc.merge_upper_chunk(upper_rgc, split_dim, threshold)
                # print('region graph chunk after merging another one: ', lower_rgc)
                
                # bbox2rgc[bbox] = lower_rgc
                # bbox2dend[bbox] = dend
                
                with open(os.path.join(rgc_dir, f'{bbox.to_filename()}.pickle'), 'wb') as f:
                    pickle.dump(lower_rgc, f)
                with open(os.path.join(rgc_dir, f'{bbox.to_filename()}.pickle'), 'rb') as f:
                    lower_rgc2 = pickle.load(f)
                assert pickle.dumps(lower_rgc) == pickle.dumps(lower_rgc2)
                # print(lower_rgc2)
                
                # print('dendrogram from inode: ', dend)
                with open(os.path.join(dend_dir, f'{bbox.to_filename()}.pickle'), 'wb') as f:
                    pickle.dump(dend, f)
                with open(os.path.join(dend_dir, f'{bbox.to_filename()}.pickle'), 'rb') as f:
                    dend2 = pickle.load(f)
                assert pickle.dumps(dend) == pickle.dumps(dend2)
                # print(dend2)

    # combined_dend = Dendrogram();
    # for _, dend in bbox2dend.items():
    #     print(dend)
    #     combined_dend.merge(dend)
    # print('combined dendrogram: ', combined_dend)
    
    print('loaded dendrogram:')
    combined_dend2 = Dendrogram()
    for file_name in os.listdir(dend_dir):
        with open(os.path.join(dend_dir, file_name), 'rb') as f:
            dend = pickle.load(f)
        # print(dend)
        combined_dend2.merge(dend)
    print('loaded combined dendrogram: ', combined_dend2)

    # seg = combined_dend.materialize(fragments, threshold)
    seg2 = combined_dend2.materialize(fragments, threshold)

    rmtree(rgc_dir)
    rmtree(dend_dir)

    return seg2

def build_fragments(affs: np.ndarray, chunk_size: tuple) -> np.ndarray:
    fragments = watershed(affs, 0, 0.9)
    # split the chunks, so the contacting surface 
    # do not have continuous segmentation id

    start_segid = 0
    for zstart in range(0, fragments.shape[0], chunk_size[0]):
        for ystart in range(0, fragments.shape[1], chunk_size[1]):
            for xstart in range(0, fragments.shape[2], chunk_size[2]):
                fragments_chunk = fragments[
                    zstart:zstart+chunk_size[0],
                    ystart:ystart+chunk_size[1],
                    xstart:xstart+chunk_size[2]]
                
                # print('fragments chunk: \n', fragments_chunk)
                fragments_chunk, seg_num = cc3d.connected_components(
                    fragments_chunk, connectivity=6, return_N=True
                )
                fragments_chunk[fragments_chunk>0] += start_segid
                start_segid += seg_num 
                fragments[
                    zstart : zstart + chunk_size[0],
                    ystart : ystart + chunk_size[1],
                    xstart : xstart + chunk_size[2]
                ] = fragments_chunk
    print('fragments: \n', fragments)
    return fragments

def evaluate_parameter_set(sz: tuple, chunk_size: tuple, threshold: float, seed: int=1):
    
    affs = get_random_affinity_map(sz, seed=seed)
    fragments = build_fragments(affs, chunk_size)
    
    print('\nsingle machine agglomeration...')
    rg = RegionGraph(affs, fragments)
    # print('region graph: ', rg)
    print('gready mean agglomeration...')
    dend = rg.greedy_merge(fragments, threshold)
    seg = dend.materialize(fragments, threshold)
    print('\nsegmentation: \n', seg)

    seg2 = distributed_agglomeration(fragments, affs, threshold, chunk_size)

    score = rand_score(seg.flatten(), seg2.flatten())
    print('rand score: ', score)
    print('fragments: \n', fragments)
    print('original segmentation: \n', seg)
    print('distributed segmentation: \n', seg2)
    assert score == 1


def test_region_graph_chunk():
    # use smaller size in debuging mode
    # so we can really check individual voxel and edges
    
    #for seed in range(10000, 900000):
    for seed in range(1):
        print(f'\nseed is {seed} \n')
        sz = (2, 6,6)
        threshold = 0.5
        chunk_size = (1, 3, 6)
        #chunk_size = (6, 3, 1)
        evaluate_parameter_set(sz, chunk_size, threshold, seed=seed)

    #threshold = 0.5
    #sz = (128,500, 40)
    #chunk_size = (50, 40, 35)
    #evaluate_parameter_set(sz, chunk_size, threshold)

