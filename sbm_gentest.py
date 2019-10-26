import Lsbm as Ls
import numpy as np
import tables as tb

from time import time
from scipy.sparse import csr_matrix

n=1000000
K=2
L=2
nk = np.array([500000,500000],dtype=np.int32)
cnk = np.array([0,500000],dtype=np.int32)
p_in = 0.0002
p_out = 0.00015
filename = 'sbmmillion15.h5'

tcomm = np.zeros(shape=(n))
tcomm[0:500000] = np.ones(shape=(500000))

h51 = tb.open_file('sbmmillion15.h5', 'r')
h52 = tb.open_file('sbmmillion.h5', 'r')

h5= [h51,h52]

labels = Ls.SpectralClusterH(h5=h5, niter=100, n=n, K=K ,bs=100000)

error = np.count_nonzero( labels-tcomm )
error1 = min(error,n-error)/n
print(error1)
#improvement
#step 1: estimate parameters
hatp = Ls.getPH(h5=h5, labels=labels,K=K,n=n,bs=100000)

#step 2: update using log-likelihood
vec_clu = Ls.greedyUpdateH(h5=h5,hatp=hatp,labels=labels, niter=5, K=K, n=n,bs=100000).transpose()

#Final result
error = np.count_nonzero( vec_clu[0,:]-tcomm )
error2 = min(error,n-error)/n
print(error2)


