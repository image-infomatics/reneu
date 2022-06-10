from typing import Union

import numpy as np
from tqdm import tqdm

from chunkflow.chunk import Chunk

from fastremap import unique

from reneu.lib.segmentation import seeded_watershed, RegionGraph


MAX_INT = 18446744073709551615

def agglomerate(affs: np.ndarray, seg: np.ndarray, 
        agglomeration_threshold: float = 0., 
        min_voxel_num_threshold: int=MAX_INT,
        max_voxel_num_threshold: int=MAX_INT):
    """
    Parameters:
    agglomeration_threshold [float]: the greedy agglomeration until a threshold.
    max_voxel_num_threshold [int]: The maximum voxel number an object can grow. 
        No object can be larger than this number.
        The default value is the maximum limit of C++ size_t
    """
    if min_voxel_num_threshold is None:
        min_voxel_num_threshold = MAX_INT
    if max_voxel_num_threshold is None:
        max_voxel_num_threshold = MAX_INT
        
    print('construct region graph...')
    rg = RegionGraph(affs, seg)
    print('gready mean agglomeration...')
    dend = rg.greedy_mean_affinity_agglomeration(
        seg, agglomeration_threshold,
        min_voxel_num_threshold=min_voxel_num_threshold, 
        max_voxel_num_threshold=max_voxel_num_threshold)
    seg = dend.materialize(seg, agglomeration_threshold)
    
    return seg



def seeded_watershed_2d(seg: np.ndarray, affs: np.ndarray, threshold: float):
    for z in tqdm(range(seg.shape[0])):
        seg2d = seg[z, :, :]
        seg2d = np.expand_dims(seg2d, 0)
        affs2d = affs[:, z, :, :]
        affs2d = np.expand_dims(affs2d, 1)

        seeded_watershed(seg2d, affs2d, threshold)
        seg[z, :, :] = seg2d


def contacting_and_inner_obj_ids(seg: Union[Chunk, np.ndarray]) -> tuple:
    """get the contacting object id set from a segmentation chunk. 
    This chunk should cover one more voxel in the neighboring chunk.
    since the affinity map is defined in the negative direction.
    As a result, the size of this chunk should be 1+block.shape.

    Args:
        seg (Union[Chunk, np.ndarray]): the segmentation chunk covering one voxel of neighboring chunk

    Returns:
        tuple: the inner object IDs and contacting seg IDs.
    """
    if isinstance(seg, Chunk):
        seg = seg.array

    uniq_neg_z = unique(seg[0:2, 1:, 1:], return_counts=False)
    uniq_neg_y = unique(seg[1:, 0:2, 1:], return_counts=False)
    uniq_neg_x = unique(seg[1:, 1:, 0:2], return_counts=False)

    uniq_pos_z = unique(seg[-1, 1:, 1:], return_counts=False)
    uniq_pos_y = unique(seg[1:, -1, 1:], return_counts=False)
    uniq_pos_x = unique(seg[1:, 1:, -1], return_counts=False)

    uniq_neg = set(uniq_neg_z).union(set(uniq_neg_y)).union(set(uniq_neg_x))
    uniq_pos = set(uniq_pos_z).union(set(uniq_pos_y)).union(set(uniq_pos_x))
    contacting = uniq_neg.union(uniq_pos)

    inner = unique(seg, return_counts=False)
    inner = set(inner)
    inner -= contacting
    contacting.discard(0)

    if inner is not None:
        assert 0 not in inner 
    
    return contacting, inner


