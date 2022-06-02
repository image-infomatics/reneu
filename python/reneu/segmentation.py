from typing import Union

import numpy as np
from tqdm import tqdm

from chunkflow.chunk import Chunk

from fastremap import unique

from reneu.lib.segmentation import seeded_watershed, RegionGraph


def agglomerate(affs: np.ndarray, seg: np.ndarray, agglomeration_threshold: float = 0., 
        voxel_num_threshold: int=18446744073709551615):
    """
    Parameters:
    voxel_num_threshold [int]: the default value is the maximum limit of C++ size_t
    """
    print('construct region graph...')
    rg = RegionGraph(affs, seg)

    print('gready mean agglomeration...')
    dend = rg.greedy_mean_affinity_agglomeration(seg, agglomeration_threshold, voxel_num_threshold)
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


def split_consensus(seg0: np.ndarray, seg1: np.ndarray) -> np.ndarray:
    """given two segmentations, get the split consensus.
    construct the contingency table. Each non-zero item in that table is 
    a new object in the split consensus.

    Args:
        seg0 (np.ndarray): the first segmentation volume
        seg1 (np.ndarray): the second segmentation volume

    Raises:
        NotImplementedError: _description_

    Returns:
        np.ndarray: the split consensus
    """
    assert np.issubdtype(seg0.dtype, np.integer)
    assert np.issubdtype(seg1.dtype, np.integer)
    assert seg0.dtype == seg1.dtype
    assert seg0.shape == seg1.shape

    raise NotImplementedError('will be done later.')