from reneu.lib.segmentation import Dendrogram


def test_dendrogram():
    dend1 = Dendrogram(0.5)
    dend1.push_edge(1,2,0.1)

    dend2 = Dendrogram(0.3)
    dend2.push_edge(2,3,0.4)

    dend1.merge(dend2)
    print('dendrogram after merging:')
    dend1.print()
