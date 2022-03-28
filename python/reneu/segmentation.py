import numpy as np
from tqdm import tqdm

from reneu.lib.segmentation import seeded_watershed


def seeded_watershed_2d(seg: np.ndarray, affs: np.ndarray, threshold: float):
    for z in tqdm(range(seg.shape[0])):
        seg2d = seg[z, :, :]
        seg2d = np.expand_dims(seg2d, 0)
        affs2d = affs[:, z, :, :]
        affs2d = np.expand_dims(affs2d, 1)

        seeded_watershed(seg2d, affs2d, threshold)
        seg[z, :, :] = seg2d

    
