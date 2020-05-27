import faulthandler
faulthandler.enable()

import numpy as np
from numpy.testing import assert_allclose
from reneu.libreneu import pca_first_component
from math import isclose
from sklearn.decomposition import PCA

def python_pca(X):
    pca = PCA(n_components=1)
    pca.fit(X)
    component = pca.components_[0]
    return component

def test_pca():
    X = np.asarray(range(0, 12), dtype=np.float32).reshape(4,3)
    # this is the result of above code
    component = np.array([-0.5773503 , -0.5773502 , -0.57735026], dtype=np.float32)
    # somehow the sklearn component are all negative
    # inverting them should not change the result of orientation
    # since there is no direction
    component = -component

    component2 = pca_first_component( X )
    assert_allclose(component, component2, rtol=1e-4)
    # the vector should already be normalized
    assert isclose(np.linalg.norm( component2 ), 1, rel_tol=1e-4)

    X = np.array(  [[261680., 162640., 790605.],
                    [261680.,  162640.,  790740.],
                    [261680.,  162640.,  790695.],
                    [261680.,  162640.,  791010.],
                    [261680.,  162640.,  790965.],
                    [261680.,  162640.,  790920.],
                    [261680.,  162640.,  790875.],
                    [261680.,  162640.,  791055.],
                    [261680.,  162640.,  790830.],
                    [261680.,  162640.,  790785.],
                    [261680.,  162640.,  790650.],
                    [261680.,  162640.,  790470.],
                    [261680.,  162480.,  790380.],
                    [261680.,  161840.,  789750.],
                    [261680.,  162560.,  790425.],
                    [261680.,  162640.,  790515.],
                    [261680.,  161920.,  789840.],
                    [261760.,  162000.,  789885.],
                    [261680.,  161840.,  789795.],
                    [261760.,  162000.,  789930.]])
    print('shape of input array: ', X.shape)
    # component = python_pca(X)
    component1 = np.asarray([ 0.02229965, -0.57740044, -0.81615652 ])
    print('python pca output: ', component1)
    component2 = pca_first_component(X)
    component3 = pca_first_component(X)
    component4 = pca_first_component(X)
    np.testing.assert_equal(component2, component3)
    np.testing.assert_equal(component2, component4)
    # print('out pca output: ', pca_first_component(X))
    np.testing.assert_almost_equal( -component1,  component2, decimal=5)

