from typing import Union 
import numpy as np
from .lib.xiuli import XSkeleton


class Skeleton(XSkeleton):
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
    def __init__(self, *args): 
        super().__init__(*args)
   
    @classmethod
    def from_nodes_and_parents(cls, nodes: np.ndarray, parents: np.ndarray, 
                    classes: np.ndarray=None):
        assert nodes.shape[1] == 4
        assert nodes.shape[0] == len(parents) == len(classes)

        node_num = nodes.shape[0]
        nodes = nodes.astype(np.float32) 

        attributes = np.zeros((node_num, 4), dtype=np.int32) - 2
        # the parents, first child and siblings should be missing initially. 
        # The zero will all point to the first node.
        if classes is not None:
            attributes[:, 0] = classes
        else:
            # default should be undefined
            attributes[:, 0] = 0

        attributes[:, 1] = parents

        return cls(nodes, attributes) 

    @classmethod
    def from_swc(cls, file_name: str):
        """
        Parameters:
        ------------
        file_name: the swc file path
        sort_id: The node index could be unsorted in some swc files, 
            we can drop the node index column after order it. Our future
            analysis assumes that the nodes are ordered.
        """
        # numpy load text is faster than my c++ implementation!
        # it might use memory map internally
        swc_array = np.loadtxt(file_name, dtype=np.float32)
        return cls( swc_array )