import numpy as np
import Lsbm as Ls

from scipy.sparse import csr_matrix


#Parameters
n = 20000     # number of nodes
K = 2         # number of communities
L = 1         # number of labels
alpha = np.array([0.5, 0.5]) #community size
T = 400000

"""
Compute variables from the parameters
nk: number of nodes for each cluster
cnk[i]: cumulative number of nodes until cluster number i
tcomm: target community information (tcomm[i] indicates the ground truth community name of item i)
"""

nk = np.zeros((K),dtype=np.int)
cnk = np.zeros((K),dtype=np.int)
tcomm = np.zeros((n),dtype=np.int8)

for i in range(0,K-1):
    nk[i] = int(alpha[i]*n)
    cnk[i+1] = cnk[i] + nk[i]
    tcomm[cnk[i]:cnk[i+1]] = i*np.ones((nk[i]),dtype=np.int)

nk[K-1] = n- cnk[K-1]
tcomm[cnk[K-1]:n] = (K-1)*np.ones((nk[K-1]),dtype=np.int)

"""
Start simulations
sim_time : number of simulations
"""
sim_time = 20
num_sim = 3
errors = np.zeros(shape=(num_sim,3,sim_time))
for case in range(0,num_sim):
    T = 700000 + 50000*case
    P = np.array([[0.5, 0.1], [0.001, 0.001]])  #connection probability
    for sim in range(0,sim_time):
        #Create Adjacency Matrices with Labels
        AmatrixP,AmatrixN = Ls.genAmatrixT(n=n, p_in = 0.5, p_out = 0.1,T = T)
        labels = Ls.SpectralCluster(matrix= AmatrixP , niter = 100, n=n, K=K )

        error = np.count_nonzero( labels-tcomm )
        error1 = min(error,n-error)/n
        errors[case,0,sim] += error1

        #improvement
        #step 1: estimate parameters
        hatp = Ls.getPT(AmatrixP,AmatrixN, labels= labels)

        #step 2: update using log-likelihood
        vec_clu = Ls.greedyUpdateT(AmatrixP,AmatrixN, hatp=hatp, labels= labels, niter=10, n=n).transpose()

        #Final result
        error = np.count_nonzero( vec_clu[0,:]-tcomm )
        error1 = min(error,n-error)/n
        errors[case,1,sim] += error1
        errors[case,2,:] = np.exp(-(T/n)*Ls.DLP(p=P[0,0],q=P[0,1]))

 


#Target Error
