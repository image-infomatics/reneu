import pandas as pd
import numpy as np

file_name = 'smat_fcwb'
file_name = 'zfish_score_table'


csv = pd.read_csv( '{}.csv'.format(file_name) )
arr = csv.to_numpy()
arr = arr[:, 1:]
np.save('{}.npy'.format(file_name), arr)
