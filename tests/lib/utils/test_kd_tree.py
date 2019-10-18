import faulthandler
faulthandler.enable()
import numpy as np

from reneu.lib.libxiuli import XThreeDTree

def test_kd_tree():
    arr = np.asarray(range(24)).reshape(8,3)
    kdtree = XThreeDTree(arr, 2)
    print(arr)

    query_node = np.asarray([6.1, 7.1, 8.1], dtype=np.float32)
    nearest_node_index = kdtree.find_nearest_k_node_indices(query_node, 1)
    print('nearest node index: ', nearest_node_index)
    assert nearest_node_index == 2
    # from scipy.spatial import KDTree
    # tree = KDTree(arr, leafsize=2)
    # tree.query(query_node)
    # returns (0.17320519087814998, 2)

    query_node = np.asarray([21.1, 23.1, 34.1], dtype=np.float32)
    nearest_node_index = kdtree.find_nearest_k_node_indices(query_node, 1)
    print('nearest node index: ', nearest_node_index)
    assert nearest_node_index == 7
    # nearest_node = kdtree.find_nearest_k_node(query_node, 1)
    # print('nearest node: ', nearest_node)
    
    query_node = np.asarray([21.1, 23.1, 34.1], dtype=np.float32)
    nearest_node_indices = kdtree.find_nearest_k_node_indices(query_node, 2)
    print('nearest node index: ', nearest_node_indices)
    assert np.all( np.asarray([6, 7]) == nearest_node_indices )