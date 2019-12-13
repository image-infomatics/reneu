import numpy as np

def raw2normalized(score_matrix: np.ndarray):
    N = score_matrix.shape[0]
    self_score = score_matrix[range(N), range(N)]
    np.testing.assert_array_equal( self_score, np.max(score_matrix, 0) )
    normalized_score_matrix = score_matrix / self_score
    assert np.max(normalized_score_matrix) == 1
    return normalized_score_matrix

def normalized2mean(normalized_score_matrix: np.ndarray):
    assert normalized_score_matrix.max() == 1
    return (normalized_score_matrix + normalized_score_matrix.transpose()) / 2
