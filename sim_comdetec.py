import numpy as np
import Lsbm as Ls

from scipy.sparse import csr_matrix


#Parameters
n = 200000     # number of nodes
K = 2         # number of communities
L = 1         # number of labels
P = np.array([[0.01, 0.007], [0.007, 0.01]])  #connection probability
alpha = np.array([0.5, 0.5]) #community size


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
sim_time = 10
num_sim = 8
errors = np.zeros(shape=(num_sim,3,sim_time))
for case in range(0,num_sim):
    P = np.array([[0.001, 0.0009-case*0.0001], [0.001, 0.001]])  #connection probability
    for sim in range(0,sim_time):
        #Create Adjacency Matrices with Labels
        Amatrix = []
        for l in range(0,L):
            Amatrix.append(Ls.genAmatrix(nk=nk,cnk=cnk,p_in=P[l,0],p_out = P[l,1],K=K))
        #Spectral Method
        Rmatrix = csr_matrix((n,n))
        r_weight = np.random.rand(L)
        for l in range(0,L):
            Rmatrix = Rmatrix + Amatrix[l]*r_weight[l]
        #Rmatrix = Amatrix[0] - Amatrix[1]
        #Spectral Algorithm
        labels = Ls.SpectralCluster(matrix= Rmatrix , niter = 100, n=n, K=K )

        error = np.count_nonzero( labels-tcomm )
        error1 = min(error,n-error)/n
        errors[case,0,sim] += error1

        #improvement
        #step 1: estimate parameters
        hatp = Ls.getP(Amatrix=Amatrix, labels= labels, L=L, K=K)

        #step 2: update using log-likelihood
        vec_clu = Ls.greedyUpdate(Amatrix = Amatrix, hatp=hatp, labels= labels, niter=10, K=K, L=L, n=n).transpose()

        #Final result
        error = np.count_nonzero( vec_clu[0,:]-tcomm )
        error1 = min(error,n-error)/n
        errors[case,1,sim] += error1

    errors[case,2,:] = np.exp(-(n/2)*Ls.DLP(p=P[0,0],q=P[0,1]))

"""
f = open('test1.txt','w')
for case in range(0,num_sim):
    f.write("case1: (%f,%f,%f,%f,%f)\n" %errors[case,:])

f.close()
"""

#Target Error
