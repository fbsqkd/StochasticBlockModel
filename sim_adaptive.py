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
num_sim = 6
errors = np.zeros(shape=(num_sim,6,sim_time))
for case in range(0,num_sim):
    T = 400000 + 50000*case
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

        #Create Adjacency Matrices with Labels for Adaptive
        AmatrixP,AmatrixN = Ls.genAmatrixT(n=n, p_in = 0.5, p_out = 0.1,T = int(0.45*T))
        labels = Ls.SpectralCluster(matrix= AmatrixP , niter = 100, n=n, K=K )

        #improvement
        #step 1: estimate parameters
        hatp = Ls.getPT(AmatrixP,AmatrixN, labels= labels)
        hatp_in = (hatp[0,0]+ hatp[1,1])/2
        hatp_out = (hatp[0,1]+ hatp[1,0])/2
        hatp[0,0] = hatp_in
        hatp[1,1] = hatp_in
        hatp[1,0] = hatp_out
        hatp[0,1]= hatp_out
        
        #step 2: update using log-likelihood
        vec_clu = Ls.greedyUpdateT(AmatrixP,AmatrixN, hatp=hatp, labels= labels, niter=10, n=n).transpose()

        #Final result
        error = np.count_nonzero( vec_clu[0,:]-tcomm )
        error1 = min(error,n-error)/n
        errors[case,3,sim] += error1

        labelsc,labelsr = np.nonzero(vec_clu) 
        hatp = Ls.getPT(AmatrixP,AmatrixN, labels= labelsc)
        hatp_in = (hatp[0,0]+ hatp[1,1])/2
        hatp_out = (hatp[0,1]+ hatp[1,0])/2
        hatp[0,0] = hatp_in
        hatp[1,1] = hatp_in
        hatp[1,0] = hatp_out
        hatp[0,1]= hatp_out

        #improvement
        indices1 = np.nonzero(vec_clu[0,:])[0]
        indices2 = np.nonzero(vec_clu[1,:])[0]
        AmatrixP2,AmatrixN2 = Ls.genAmatrix2T(indices1=indices1,indices2=indices2, p_in = 0.5, p_out = 0.1,T = int(0.45*T))

        AmatrixP = AmatrixP+AmatrixP2
        AmatrixN = AmatrixN+AmatrixN2

        Threshold = (hatp[0,0]*(np.log(hatp[0,0])-np.log(hatp[1,0]))+(1-hatp[0,1])*(np.log(1-hatp[0,0])-np.log(1-hatp[1,0]))  )*(T/n)
        vec_clu2, rem_vec2 = Ls.greedyUpdate2T(AmatrixP,AmatrixN,indices1=indices1,indices2=indices2, hatp=hatp, Thr=Threshold, clu_vec= vec_clu.transpose(), n=n)

        error = np.count_nonzero( (vec_clu2[:,0]+rem_vec2[:,0]).transpose()-tcomm )
        error1 = min(error,n-error)/n
        errors[case,4,sim] += error1

        indices1 = np.nonzero(vec_clu2[:,0])[0]
        indices2 = np.nonzero(vec_clu2[:,1])[0]
        indices3 = np.nonzero(rem_vec2)[0]

        AmatrixP2,AmatrixN2 = Ls.genAmatrix3T(n=n, indices1=indices1,indices2=indices2,indices3=indices3, p_in = 0.5, p_out = 0.1,T = int(0.1*T))

        
        AmatrixP = AmatrixP+AmatrixP2
        AmatrixN = AmatrixN+AmatrixN2

        vec_clu2 = Ls.greedyUpdate3T(AmatrixP,AmatrixN,indices1=indices1,indices2=indices2,indices3=indices3, hatp=hatp, clu_vec= vec_clu2, n=n)

        error = np.count_nonzero( vec_clu2[:,0].transpose()-tcomm )
        error1 = min(error,n-error)/n
        errors[case,5,sim] += error1

        
        



#Target Error
