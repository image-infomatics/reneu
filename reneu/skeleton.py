from typing import Union 

import numpy as np


class Skeleton():
    """Neuron skeleton 
    
    Parameters
    ----------
    nodes: (float ndarray, node_num x 4), each row is a node with r,z,y,x 
    parents: (int ndarray, node_num), the parent node index of each node
    types: (int ndarray, node_num), the type of each node.
        The type of node is defined in `SWC format
        <http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_:

        0 - undefined
        1 - soma
        2 - axon
        3 - (basal) dendrite
        4 - apical dendrite
        5 - fork point
        6 - end point
        7 - custom
    """
    def __init__(self, nodes: np.ndarray, parents: np.ndarray, 
                    classes: np.ndarray=None):
        assert nodes.shape[1] == 4
        assert nodes.shape[0] == len(parents) == len(classes)

        self.nodes = nodes 
        
        # the parents, first child and siblings should be missing initially. 
        # The zero will all point to the first node.
        self.attributes[:, 1:] = -1
        if classes:
            self.attributes[:, 0] = classes
        self.attributes[:, 1] = parents
        self._update_child_and_sibling() 

    def _update_child_and_sibling(self):
        

    @classmethod
    def from_swc(cls, file_name: str, sort_id: bool = True):
        """
        Parameters:
        ------------
        file_name: the swc file path
        sort_id: The node index could be unsorted in some swc files, 
            we can drop the node index column after order it. Our future
            analysis assumes that the nodes are ordered.
        """
        data = np.loadtxt(file_name)
        if sort_id:
            data = data[data[:, 0].argsort()]

        # the swc file stores x,y,z,r 
        # we are defining node array as r,z,y,x to be consistent with numpy indexing
        nodes = np.fliplr(data[:, 2:6]).astype(np.float)

        # numpy index is from zero rather than 1
        parents = data[:, 7] - 1

        cls(nodes, parents, classes=data[:, 1])

    @property
    def node_num(self):
        self.nodes.shape[0]


if __name__ == '__main__':
    from time import process_time
    from os.path import join
    start = process_time()
    skeleton = Skeleton.from_swc(join(__file__, '../data/Nov10IR3e.CNG.swc'))
    print('reading time elapse: ', process_time() - start)
