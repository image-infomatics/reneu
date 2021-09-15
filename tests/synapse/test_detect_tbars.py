import numpy as np
import fill_voids
from edt import edt
from reneu.lib.synapse import detect_points, get_object_average_intensity

def test_detect_tbars():
    seg = np.zeros((7, 7, 7), dtype=bool)
    seg[2:5, 2:5, 2:5] = True

    df = edt(seg)
    seg = seg.astype(np.uint64)
    points = detect_points(seg, df)
    # print('points: ', points)
    np.testing.assert_array_equal(points, np.asarray([[3,3,3]], dtype=np.uint64))

    average_intensity = get_object_average_intensity(seg, df)
    # print('average intensity: ', average_intensity)

    avg = np.sum(df[seg>0]) / np.count_nonzero(seg)
    np.testing.assert_array_almost_equal(average_intensity, np.asarray([avg], dtype=np.float32))



# def test_fill_voids():
#     seg[2, 3, 4] = False
#     fill_voids.fill(seg, in_place=True)
#     assert seg[2,3,4] == True

def test_detect_tbars_non_symmetric():

    seg = np.zeros((7, 7, 7), dtype=bool)
    seg[1:4, 2:5, 3:6] = True
    
    # seg[2, 3, 4] = False
    # fill_voids.fill(seg, in_place=True)
    # assert seg[2,3,4] == True

    df = edt(seg)
    seg = seg.astype(np.uint64)
    points = detect_points(seg, df)
    # print('points: ', points)
    np.testing.assert_array_equal(points, np.asarray([[2,3,4]], dtype=np.uint64))

    average_intensity = get_object_average_intensity(seg, df)
    avg = np.sum(df[seg>0]) / np.count_nonzero(seg)
    np.testing.assert_array_almost_equal(average_intensity, np.asarray([avg], dtype=np.float32))


