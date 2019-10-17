import faulthandler
faulthandler.enable()

import numpy as np
from numpy.testing import assert_allclose
from reneu.lib import libxiuli
from math import isclose
#from sklearn.decomposition import PCA


def test_pca():
    X = np.asarray(range(0, 12), dtype=np.float32).reshape(4,3)
    # pca = PCA(n_components=1)
    # pca.fit(X)
    # component = pca.components_[0]
    # this is the result of above code
    component = np.array([-0.5773503 , -0.5773502 , -0.57735026], dtype=np.float32)
    # somehow the sklearn component are all negative
    # inverting them should not change the result of orientation
    # since there is no direction
    component = -component

    component2 = libxiuli.pca_first_component( X )
    assert_allclose(component, component2, rtol=1e-4)
    # the vector should already be normalized
    assert isclose(np.linalg.norm( component2 ), 1, rel_tol=1e-4)