from typing import Union
from copy import deepcopy
from math import ceil

from collections import defaultdict

import numpy as np


from cloudvolume.lib import Bbox, Vec



class BinaryBoundingBoxTree:
    def __init__(self, bbox: Bbox, leaf_size: Union[tuple, list], 
                    boundary_flag: list = [True, True, True, True, True, True]):
        """
        This tree split the segmentation volume to subvolumes.
        The algorithm is inspired by KD tree.

        bbox: the volume included in this node.
        leaf_size: the size of the mip 0 chunk size.
        boundary_flag: whether this face is a part of volume bounday. 
            The objects touching volume boundary should not be freezed.
            The flags represent: [-z, -y, -x, +z, +y, +x] directions.
        """
        assert bbox.ndim == 3
        assert len(leaf_size) == 3
        assert len(boundary_flag) == 6
        
        self.bbox = bbox
        self.boundary_flag = boundary_flag
        self.order = None

        # this node should be splitted
        # find the relatively longest axis to split
        volume_size = self.bbox.size()
        relative_length = [vs/ls for vs, ls in zip(volume_size, leaf_size)]
        if np.all([rl<=1 for rl in relative_length]):
            # this is a leaf node 
            self.lower = None
            self.upper = None
            self.split_dim = None
        else:
            self.split_dim = int(np.argmax(relative_length))
            assert volume_size[self.split_dim] > leaf_size[self.split_dim]
            lower_size = deepcopy(volume_size)
            # round up to incorporate the case of a little bit larger than leaf size
            lower_size[self.split_dim] = ceil( \
                lower_size[self.split_dim]/leaf_size[self.split_dim]) \
                //2*leaf_size[self.split_dim]
            lower_bbox = Bbox.from_delta(bbox.minpt, lower_size)
            lower_boundary_flag = deepcopy(boundary_flag)
            lower_boundary_flag[self.split_dim + 3] = False 
            self.lower = BinaryBoundingBoxTree(lower_bbox, leaf_size, boundary_flag=lower_boundary_flag)
            
            offset = Vec(0,0,0)
            offset[self.split_dim] += lower_size[self.split_dim]
            upper_minpt = bbox.minpt + offset
            upper_bbox = Bbox.from_list([*upper_minpt, *bbox.maxpt])
            upper_boundary_flag = deepcopy(boundary_flag)
            upper_boundary_flag[self.split_dim] = False
            self.upper = BinaryBoundingBoxTree(upper_bbox, leaf_size, boundary_flag=upper_boundary_flag)

    @property
    def is_leaf(self):
        return self.split_dim is None

    @property
    def shape(self):
        return self.bbox.size()
    
    @property
    def size(self):
        return self.bbox.size()

    @property
    def contact_bbox(self):
        offset = [0,0,0]
        offset[self.split_dim] = -1
        minpt = self.upper.bbox.minpt + offset
        offset[self.split_dim] = 1
        maxpt = self.lower.bbox.maxpt + offset
        return Bbox.from_list(*minpt, *maxpt)

    @property
    def node_order(self):
        """
        the leaf node order is 0
        the leaf node's parent node order is 1
        The parent node order is max(lower, upper)+1
        """
        if self.order is None:
            if self.is_leaf:
                self.order = 0
            else:
                self.order = max(self.lower.node_order, self.upper.node_order)+1
        return self.order
    
    @property
    def order2tasks(self):
        ret = defaultdict(dict)
        if self.is_leaf:
            task = (self.boundary_flag, self.split_dim, None, None)
        else:
            task = (self.boundary_flag, self.split_dim, self.lower.bbox, self.upper.bbox)
        ret[self.node_order][self.bbox] = task

        if not self.is_leaf:
            lower_tasks = self.lower.order2tasks
            upper_tasks = self.upper.order2tasks
            for order, tasks in lower_tasks.items():
                # we can use x | y to merge dict in python 3.9
                ret[order] = {**ret[order], **tasks}
            for order, tasks in upper_tasks.items():
                ret[order] = {**ret[order], **tasks}
        assert len(ret) == self.order + 1
        return ret
        