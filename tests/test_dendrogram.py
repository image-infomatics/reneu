import pickle

from reneu.lib.segmentation import Dendrogram


def test_dendrogram():
    dend1 = Dendrogram(0.5)
    dend1.push_edge(1,2,0.1)

    dend2 = Dendrogram(0.3)
    dend2.push_edge(2,3,0.4)

    dend1.merge(dend2)
    print('dendrogram after merging:')
    dend1.print()

    print('test serialization...')
    data = pickle.dumps(dend1)
    # print('bytes of dendrogram 1 : ', data)
    dend3 = pickle.loads(data)
    data3 = pickle.dumps(dend3)
    # print('bytes of dendrogram 3: ', data3)
    assert data == data3
    # dend3.print() 
