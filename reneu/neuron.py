import numpy as np
from .skeleton import Skeleton

class Neuron(Skeleton):
    def __init__(self, nodes: np.ndarray, parents: np.ndarray, 
                    classes: np.ndarray=None, synapses: np.ndarray=None):
        super().__init__(nodes, parents, classes=classes)
