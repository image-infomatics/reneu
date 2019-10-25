import faulthandler
faulthandler.enable()

import numpy as np
from scipy.spatial import KDTree

from reneu.lib.libxiuli import XThreeDTree

def get_nearest_node_indices(nodes, query_node, k=1, leaf_size=20):
    tree = KDTree(nodes, leafsize=leaf_size)
    res = tree.query(query_node, k=k)
    return res[1]

# def get_nearest_node_indices2(nodes, query_node, k=1, leaf_size=20):
    # from sklearn.neighbors import KDTree as SKKDTree
    # tree = SKKDTree(nodes, leaf_size=leaf_size)
    # query_node = np.expand_dims(query_node, axis=0)
    # _, ind = tree.query(query_node, k=k)
    # return ind 

def test_kd_tree():
    leaf_size = 2
    nodes = np.asarray(range(24)).reshape(8,3)
    kdtree = XThreeDTree(nodes, leaf_size)
    print(nodes)

    query_node = np.asarray([6.1, 7.1, 8.1], dtype=np.float32)
    nearest_node_index = kdtree.find_nearest_k_node_indices(query_node, 1)
    print('nearest node index: ', nearest_node_index)
    assert nearest_node_index == 2

    query_node = np.asarray([21.1, 23.1, 34.1], dtype=np.float32)
    nearest_node_index = kdtree.find_nearest_k_node_indices(query_node, 1)[0]
    print('nearest node index: ', nearest_node_index)
    assert nearest_node_index == 7
    # nearest_node = kdtree.find_nearest_k_node(query_node, 1)
    # print('nearest node: ', nearest_node)
    
    query_node = np.asarray([21.1, 23.1, 34.1], dtype=np.float32)
    nearest_node_indices = kdtree.find_nearest_k_node_indices(query_node, 2)
    print('nearest node index: ', nearest_node_indices)
    assert np.all( np.asarray([6, 7]) == nearest_node_indices )
    
    print('\ntest nearest node number more than leaf node number.')
    query_node = np.asarray([21.1, 23.1, 34.1], dtype=np.float32)
    nearest_node_indices = kdtree.find_nearest_k_node_indices(query_node, 3)
    true_nearest_node_indices = get_nearest_node_indices(
                                    nodes, query_node, k=3, leaf_size=leaf_size)
    print('nearest node index: ', nearest_node_indices)
    print('true nearest node index: ', true_nearest_node_indices)
    assert len(set(nearest_node_indices).symmetric_difference(set(true_nearest_node_indices)))==0

    print('\ntest leaf node number is more than nearest node number.') 
    kdtree = XThreeDTree(nodes, 3)
    query_node = np.asarray([21.1, 23.1, 34.1], dtype=np.float32)
    nearest_node_indices = kdtree.find_nearest_k_node_indices(query_node, 2)
    print('nearest node index: ', nearest_node_indices)
    assert np.all( np.asarray([6, 7]) == nearest_node_indices )


def test_large_fake_array():
    print('\nlarger fake array test')
    node_num = 20
    leaf_size = 2
    query_index = 2
    nodes = np.zeros((node_num, 3), dtype=np.float32)
    nodes[:, 0] = np.arange(0, node_num)
    nodes[:, 1] = np.arange(0, node_num)
    nodes[:, 2] = np.arange(0, node_num)
    kdtree = XThreeDTree(nodes, leaf_size)
    query_node = np.asarray([query_index+0.1, query_index+0.1, query_index+0.1], 
                                                                    dtype=np.float32)
    nearest_node_index = kdtree.find_nearest_k_node_indices(query_node, 1)[0]
    true_nearest_node_index = get_nearest_node_indices(nodes, query_node, 
                                                            k=1, leaf_size=leaf_size)

    print('nearest node index: ', nearest_node_index)
    print('true nearest node index: ', true_nearest_node_index)
    assert true_nearest_node_index == nearest_node_index

    k = 20
    nearest_node_indices = kdtree.find_nearest_k_node_indices(query_node, k)
    true_nearest_node_indices = get_nearest_node_indices(nodes, query_node, 
                                                            k=k, leaf_size=leaf_size)
    # true_nearest_node_indices2 = get_nearest_node_indices2(nodes, query_node, 
                                                            # k=k, leaf_size=leaf_size)

    print(f'\nnearest {k} node indices: {nearest_node_indices}')
    print(f'\ntrue nearest {k} node indices: {true_nearest_node_indices}')
    assert len(set(nearest_node_indices).symmetric_difference(set(true_nearest_node_indices)))==0
    # print(f'\ntrue nearest {k} node indices2: {true_nearest_node_indices2}')

if __name__ == '__main__':
    test_large_fake_array()