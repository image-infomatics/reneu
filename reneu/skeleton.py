from typing import Union 
import numpy as np
from .lib.libreneu import update_first_child_and_sibling, downsample, skeleton_to_swc_string


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
    def __init__(self, nodes: np.ndarray, attributes: np.ndarray, 
                    force_update_child_and_sibling: bool = True):
        assert nodes.shape[1] == 4
        assert attributes.shape[1] == 4

        # make sure that the child and sibling are consistent with the parents
        if force_update_child_and_sibling:
            update_first_child_and_sibling(attributes)

        self.nodes = nodes 
        self.attributes = attributes
    
    @classmethod
    def from_nodes_and_parents(cls, nodes: np.ndarray, parents: np.ndarray, 
                    classes: np.ndarray=None):
        assert nodes.shape[1] == 4
        assert nodes.shape[0] == len(parents) == len(classes)

        node_num = nodes.shape[0] 

        attributes = np.zeros((node_num, 4), dtype=np.float32) 
        # the parents, first child and siblings should be missing initially. 
        # The zero will all point to the first node.
        attributes[:, 1:] = -1
        if classes is not None:
            attributes[:, 0] = classes
        attributes[:, 1] = parents
    
        return cls(nodes, attributes, force_update_child_and_sibling=True)

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
        nodes = data[:, 2:6].astype(np.float32)

        # numpy index is from zero rather than 1
        parents = data[:, 6] - 1

        return cls.from_nodes_and_parents(nodes, parents, classes=data[:, 1])

    def to_swc(self, file_name: str):
        # the last argument is precision
        swc_str = skeleton_to_swc_string(self.nodes, self.attributes, 3)
        with open(file_name, 'w') as f:
            f.write(swc_str)

    @property
    def node_num(self):
        return self.nodes.shape[0]

    def __len__(self):
        return self.nodes.shape[0] 
    
    def downsample(self, step: float=1000., modify_in_place: bool = False):
        """Downsample the skeleton to node interval of step distance.
        Normally the node coordinate unit is nm, so 1000 nm step will 
        create a new skeleton with node distance about 1 micron. Note that 
        we fixed the segment starting and ending nodes, so the first and 
        last two nodes could be very close in one segment.

        Parameters
        -----------
        step: distance between nodes after downsampling
        modify_in_place: make modifycations in place to avoid memory copy. 
            This will change current skeleton and make is unusable any more!

        Return:
        -------
        ret: the downsampled skeleton.
        """
        # our downsample function changes the value of nodes and families
        if not modify_in_place:
            nodes = np.copy( self.nodes )

        nodes, attributes = downsample(self.nodes, self.attributes)
        return Skeleton(nodes, attributes)


if __name__ == '__main__':
    from time import process_time
    from os.path import join
    start = process_time()
    skeleton = Skeleton.from_swc(join(__file__, '../data/Nov10IR3e.CNG.swc'))
    print('reading time elapse: ', process_time() - start)
