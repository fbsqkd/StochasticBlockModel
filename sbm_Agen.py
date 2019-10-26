import Lsbm as Ls
import numpy as np
import tables as tb

n=1000000
K=2
nk = np.array([500000,500000],dtype=np.int32)
cnk = np.array([0,500000],dtype=np.int32)
p_in = 0.00015
p_out = 0.00013
filename = 'sbmmillion1513.h5'

Ls.genAmatrixH(nk = nk,cnk = cnk, p_in=p_in,p_out=p_out,K=K,filename = filename)
