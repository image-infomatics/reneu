import pickle
import numpy as np
np.random.seed(0)

from reneu.lib.segmentation import Dendrogram


def test_dendrogram():
    dend1 = Dendrogram(0.01)
    dend1.push_edge(1,2,0.1)

    dend2 = Dendrogram(0.3)
    dend2.push_edge(2,3,0.4)

    dend1.merge(dend2)
    print('dendrogram after merging:', dend1)
    print('as array: \n', dend1.array)

    breakpoint()
    dsets = dend1.to_disjoint_sets(0.2)
    # root = dsets.find_set(2)
    dend1.split_objects(0.1, set({2}), 0.3)

    print('test serialization...')
    data = pickle.dumps(dend1)
    # print('bytes of dendrogram 1 : ', data)
    dend3 = pickle.loads(data)
    data3 = pickle.dumps(dend3)
    # print('bytes of dendrogram 3: ', data3)
    assert data == data3

    print('test keep contacting edges...')
    seg = np.random.randint(20, dtype=np.uint64, size=(64,64,64))
    dend1.push_edge(22, 21, 0.4)
    # dend1.print()
    assert dend1.edge_num == 3
    dend1.keep_only_contacting_edges(seg, (8,8,8))
    # dend1.print()
    assert dend1.edge_num == 2