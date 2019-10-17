import faulthandler
faulthandler.enable()
import numpy as np

from reneu.lib.libxiuli import XThreeDTree

# def test_kd_tree():
arr = np.asarray(range(24)).reshape(8,3)
print(arr)
kdtree = XThreeDTree(arr, 2)
query_node = np.asarray([6.1, 7.1, 8.1], dtype=np.float32)
nearest_node_index = kdtree.find_nearest_k_node_indices(query_node, 1)
print('nearest node index: ', nearest_node_index)