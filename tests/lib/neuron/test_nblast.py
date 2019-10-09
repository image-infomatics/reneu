import os
import numpy as np
from math import isclose

from reneu.lib.libxiuli import XNBLASTScoreTable
from reneu.neuron import Skeleton
from reneu.lib.libxiuli import XVectorCloud, XNBLASTScoreMatrix

DATA_DIR = os.path.join(os.path.dirname(__file__), '../../../data/')
table_path = os.path.join(DATA_DIR, 'smat_fcwb.csv')
st = XNBLASTScoreTable( table_path )
# somehow this is not working
# st = XNBLASTScoreTable()

def test_nblast_score_table():
    assert isclose(st[0., 0.], 9.500096818, abs_tol=1e-4)
    assert isclose(st[50000, 0.93], -10.1287588679926, abs_tol=1e-4)
    assert isclose(st[26000, 0.73], -2.70184701924541, abs_tol=1e-4)
    assert isclose(st[11000, 0.62], 0.28008292141423, abs_tol=1e-4)
    # test the boundary condition
    assert isclose(st[2000, 0.4], 8.23731427922735, abs_tol=1e-4)

def test_nblast():    
    # the result from R NBLAST is :
    # ID    77625	    77641
    # 77625	86501.20 	53696.72
    # 77641	50891.03 	101011.08
    # Note that R NBLAST use micron as unit, and our NBLAST use nanometer
    # the swc here use nanometer, so no need to divide by 1000
    for i in range(200):
        sk1 = Skeleton.from_swc( os.path.join(DATA_DIR, '77625.swc') )
        sk2 = Skeleton.from_swc( os.path.join(DATA_DIR, '77641.swc') )

        cv1 = XVectorCloud( sk1.nodes, 20 )
        cv2 = XVectorCloud( sk2.nodes, 20 )
        score = cv1.query_by( cv2, st )
        print('nblast score: ', score)
        assert isclose( score, 50891.03, rel_tol = 1e-3)

    cvs = [ cv1, cv2 ]
    score_matrix = XNBLASTScoreMatrix(cvs, st)
    print('raw scores: ', score_matrix.raw_score_matrix)
    #breakpoint()