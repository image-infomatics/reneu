import numpy as np
from edt import edt
from reneu.lib.synapse import detect_points

def test_detect_tbars():
    seg = np.zeros((7, 7, 7), dtype=bool)
    seg[2:5, 2:5, 2:5] = True

    df = edt(seg)
    seg = seg.astype(np.uint64)
    points = detect_points(seg, df)
    # print('points: ', points)
    np.testing.assert_array_equal(points, np.asarray([[3,3,3]], dtype=np.uint64))