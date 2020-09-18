import faulthandler
faulthandler.enable()

import numpy as np
from scipy.spatial import KDTree

from reneu.lib.skeleton import XKDTree

def get_nearest_point_indices(points, query_point, k=1, leaf_size=20):
    tree = KDTree(points, leaf_size)
    res = tree.query(query_point, k=k)
    return res[1]

# def get_nearest_point_indices2(points, query_point, k=1, leaf_size=20):
    # from sklearn.neighbors import XKDTree as SKXKDTree
    # tree = SKXKDTree(points, leaf_size=leaf_size)
    # query_point = np.expand_dims(query_point, axis=0)
    # _, ind = tree.query(query_point, k=k)
    # return ind 

def test_kd_tree():
    leaf_size = 2
    points = np.asarray(range(24)).reshape(8,3)
    kdtree = XKDTree(points, leaf_size)
    print('points: ', points)

    query_point = np.asarray([6.1, 7.1, 8.1], dtype=np.float32)
    nearest_point_index = kdtree.knn(query_point, 1)
    print('nearest point index: ', nearest_point_index)
    assert nearest_point_index == 2

    query_point = np.asarray([21.1, 23.1, 34.1], dtype=np.float32)
    nearest_point_index = kdtree.knn(query_point, 1)[0]
    print('nearest point index: ', nearest_point_index)
    assert nearest_point_index == 7
    # nearest_point = kdtree.find_nearest_k_point(query_point, 1)
    # print('nearest point: ', nearest_point)
    
    query_point = np.asarray([21.1, 23.1, 34.1], dtype=np.float32)
    nearest_point_indices = kdtree.knn(query_point, 2)
    print('nearest point index: ', nearest_point_indices)
    assert np.all( np.asarray([6, 7]) == nearest_point_indices )
    
    print('\ntest nearest point number more than leaf point number.')
    query_point = np.asarray([21.1, 23.1, 34.1], dtype=np.float32)
    nearest_point_indices = kdtree.knn(query_point, 3)
    true_nearest_point_indices = get_nearest_point_indices(
                                    points, query_point, k=3, leaf_size=leaf_size)
    print('nearest point index: ', nearest_point_indices)
    print('true nearest point index: ', true_nearest_point_indices)
    assert len(set(nearest_point_indices).symmetric_difference(set(true_nearest_point_indices)))==0

    print('\ntest leaf point number is more than nearest point number.') 
    kdtree = XKDTree(points, 3)
    query_point = np.asarray([21.1, 23.1, 34.1], dtype=np.float32)
    nearest_point_indices = kdtree.knn(query_point, 2)
    print('nearest point index: ', nearest_point_indices)
    assert np.all( np.asarray([6, 7]) == nearest_point_indices )


def test_large_fake_array():
    print('\nlarger fake array test')
    point_num = 100
    leaf_size = 2
    query_index = 10
    points = np.zeros((point_num, 3), dtype=np.float32)
    points[:, 0] = np.arange(0, point_num)
    points[:, 1] = np.arange(0, point_num)
    points[:, 2] = np.arange(0, point_num)
    kdtree = XKDTree(points, leaf_size)
    query_point = np.asarray([query_index+0.1, query_index+0.1, query_index+0.1], 
                                                                    dtype=np.float32)
    nearest_point_index = kdtree.knn(query_point, 1)[0]
    true_nearest_point_index = get_nearest_point_indices(points, query_point, 
                                                            k=1, leaf_size=leaf_size)

    print('\n\nquery nearest neighbor.')
    print('nearest point index: ', nearest_point_index)
    print('true nearest point index: ', true_nearest_point_index)
    assert true_nearest_point_index == nearest_point_index

    k = 20
    print(f'\n\n query {k} nearest neighbors.')
    nearest_point_indices = kdtree.knn(query_point, k)
    true_nearest_point_indices = get_nearest_point_indices(points, query_point, 
                                                            k=k, leaf_size=leaf_size)
    # true_nearest_point_indices2 = get_nearest_point_indices2(points, query_point, 
                                                            # k=k, leaf_size=leaf_size)

    print(f'\nnearest {k} point indices: {nearest_point_indices}')
    print(f'\ntrue nearest {k} point indices: {true_nearest_point_indices}')
    assert len(set(nearest_point_indices).symmetric_difference(set(true_nearest_point_indices)))==0
    # print(f'\ntrue nearest {k} point indices2: {true_nearest_point_indices2}')

if __name__ == '__main__':
    test_large_fake_array()
