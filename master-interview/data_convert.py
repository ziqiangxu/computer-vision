import scipy.io as io
import numpy as np

mat = io.loadmat('nod.mat')
nod = mat['nod']
np.save('nod.npy', nod)

mat = io.loadmat('lung.mat')
lung = mat['vol']
np.save('lung.npy', lung)
