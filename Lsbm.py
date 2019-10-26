import numpy as np
import tables as tb

from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, rand

from sklearn.cluster import KMeans

"""
Create Adjacency Matrices
nk: k dim vector, number of nodes for each cluster
cnk: cumulative version of nk
p_in: intra conneciton probability
p_out: inter connection probability
"""
def genAmatrix(nk,cnk,p_in,p_out,K):
    n = cnk[K-1] + nk[K-1]
    
    Adjacency = csr_matrix((n,n))
    columns = np.array([] , dtype = np.int32)
    rows = np.array([], dtype = np.int32)
    datas = np.array([])
    
    for k in range(0,K):
        density = 1 - np.sqrt(1-p_in)
        con = rand(nk[k],nk[k],density = density, format = 'coo')
        con.col[:] = con.col[:] + cnk[k]
        con.row[:] = con.row[:] + cnk[k]
        columns = np.append(columns,con.col)
        rows = np.append(rows,con.row)
        datas = np.append(datas,con.data)
        for j in range(k+1,K):
            con = rand(nk[k],nk[j], density= p_out, format = 'coo')
            con.col[:] = con.col[:] + cnk[k]
            con.row[:] = con.row[:] + cnk[j]
            columns = np.append(columns,con.col)
            rows = np.append(rows,con.row)
            datas = np.append(datas,con.data)
    Adjacency = coo_matrix((datas, (rows,columns)),shape = (n,n)).tocsr()
    Adjacency = Adjacency - Adjacency.transpose()

    Adjacency.data[:] = 1

    return Adjacency

def genAmatrixT(n,p_in,p_out,T):
    
    AdjacencyP = csr_matrix((n,n))
    AdjacencyN = csr_matrix((n,n))
    columnsP = np.array([] , dtype = np.int32)
    rowsP = np.array([], dtype = np.int32)
    columnsN = np.array([] , dtype = np.int32)
    rowsN = np.array([], dtype = np.int32)
    datasP = np.array([])
    datasN = np.array([])
    cluster_size = int(n/2)
    
    for t in range(0,T):
        tcol = np.random.randint(0,n)
        trow = np.random.randint(0,n)
        if np.abs(int(tcol/cluster_size) - int(trow/cluster_size)) == 0:
            if np.random.random() < p_in:
                columnsP = np.append(columnsP, tcol)
                rowsP = np.append(rowsP, trow)
                datasP = np.append(datasP,1)
            else:
                columnsN = np.append(columnsN, tcol)
                rowsN = np.append(rowsN, trow)
                datasN = np.append(datasN,1)
        else:
            if np.random.random() < p_out:
                columnsP = np.append(columnsP, tcol)
                rowsP = np.append(rowsP, trow)
                datasP = np.append(datasP,1)
            else:
                columnsN = np.append(columnsN, tcol)
                rowsN = np.append(rowsN, trow)
                datasN = np.append(datasN,1)

    AdjacencyP = coo_matrix((datasP, (rowsP,columnsP)),shape = (n,n)).tocsr()
    AdjacencyP = AdjacencyP + AdjacencyP.transpose()

    AdjacencyN = coo_matrix((datasN, (rowsN,columnsN)),shape = (n,n)).tocsr()
    AdjacencyN = AdjacencyN + AdjacencyN.transpose()

    AdjacencyP.data[:] = 1
    AdjacencyN.data[:] = 1

    return AdjacencyP, AdjacencyN

def genAmatrix2T(indices1,indices2,p_in,p_out,T):

    n1 = len(indices1)
    n2 = len(indices2)
    n = n1+n2
    
    AdjacencyP = csr_matrix((n,n))
    AdjacencyN = csr_matrix((n,n))
    columnsP = np.array([] , dtype = np.int32)
    rowsP = np.array([], dtype = np.int32)
    columnsN = np.array([] , dtype = np.int32)
    rowsN = np.array([], dtype = np.int32)
    datasP = np.array([])
    datasN = np.array([])
    cluster_size = int(n/2)

    
    for t in range(0,int(0.5*T)):

        tcol = indices1[np.random.randint(0,n1)]
        trow = indices1[np.random.randint(0,n1)]
        
        if np.abs(int(tcol/cluster_size) - int(trow/cluster_size)) == 0:
            if np.random.random() < p_in:
                columnsP = np.append(columnsP, tcol)
                rowsP = np.append(rowsP, trow)
                datasP = np.append(datasP,1)
            else:
                columnsN = np.append(columnsN, tcol)
                rowsN = np.append(rowsN, trow)
                datasN = np.append(datasN,1)
        else:
            if np.random.random() < p_out:
                columnsP = np.append(columnsP, tcol)
                rowsP = np.append(rowsP, trow)
                datasP = np.append(datasP,1)
            else:
                columnsN = np.append(columnsN, tcol)
                rowsN = np.append(rowsN, trow)
                datasN = np.append(datasN,1)
                
        tcol = indices2[np.random.randint(0,n2)]
        trow = indices2[np.random.randint(0,n2)]
        if np.abs(int(tcol/cluster_size) - int(trow/cluster_size)) == 0:
            if np.random.random() < p_in:
                columnsP = np.append(columnsP, tcol)
                rowsP = np.append(rowsP, trow)
                datasP = np.append(datasP,1)
            else:
                columnsN = np.append(columnsN, tcol)
                rowsN = np.append(rowsN, trow)
                datasN = np.append(datasN,1)
        else:
            if np.random.random() < p_out:
                columnsP = np.append(columnsP, tcol)
                rowsP = np.append(rowsP, trow)
                datasP = np.append(datasP,1)
            else:
                columnsN = np.append(columnsN, tcol)
                rowsN = np.append(rowsN, trow)
                datasN = np.append(datasN,1)
        
    AdjacencyP = coo_matrix((datasP, (rowsP,columnsP)),shape = (n,n)).tocsr()
    AdjacencyP = AdjacencyP + AdjacencyP.transpose()

    AdjacencyN = coo_matrix((datasN, (rowsN,columnsN)),shape = (n,n)).tocsr()
    AdjacencyN = AdjacencyN + AdjacencyN.transpose()

    AdjacencyP.data[:] = 1
    AdjacencyN.data[:] = 1

    return AdjacencyP, AdjacencyN

def genAmatrix3T(n,indices1,indices2,indices3, p_in,p_out,T):

    n1 = len(indices1)
    n2 = len(indices2)
    n3 = len(indices3)
    
    AdjacencyP = csr_matrix((n,n))
    AdjacencyN = csr_matrix((n,n))
    columnsP = np.array([] , dtype = np.int32)
    rowsP = np.array([], dtype = np.int32)
    columnsN = np.array([] , dtype = np.int32)
    rowsN = np.array([], dtype = np.int32)
    datasP = np.array([])
    datasN = np.array([])
    cluster_size = int(n/2)

    
    for t in range(0,int(0.5*T)):

        tcol = indices3[np.random.randint(0,n3)]
        trow = indices1[np.random.randint(0,n1)]
        
        if np.abs(int(tcol/cluster_size) - int(trow/cluster_size)) == 0:
            if np.random.random() < p_in:
                columnsP = np.append(columnsP, tcol)
                rowsP = np.append(rowsP, trow)
                datasP = np.append(datasP,1)
            else:
                columnsN = np.append(columnsN, tcol)
                rowsN = np.append(rowsN, trow)
                datasN = np.append(datasN,1)
        else:
            if np.random.random() < p_out:
                columnsP = np.append(columnsP, tcol)
                rowsP = np.append(rowsP, trow)
                datasP = np.append(datasP,1)
            else:
                columnsN = np.append(columnsN, tcol)
                rowsN = np.append(rowsN, trow)
                datasN = np.append(datasN,1)
                
        tcol = indices3[np.random.randint(0,n3)]
        trow = indices2[np.random.randint(0,n2)]
        if np.abs(int(tcol/cluster_size) - int(trow/cluster_size)) == 0:
            if np.random.random() < p_in:
                columnsP = np.append(columnsP, tcol)
                rowsP = np.append(rowsP, trow)
                datasP = np.append(datasP,1)
            else:
                columnsN = np.append(columnsN, tcol)
                rowsN = np.append(rowsN, trow)
                datasN = np.append(datasN,1)
        else:
            if np.random.random() < p_out:
                columnsP = np.append(columnsP, tcol)
                rowsP = np.append(rowsP, trow)
                datasP = np.append(datasP,1)
            else:
                columnsN = np.append(columnsN, tcol)
                rowsN = np.append(rowsN, trow)
                datasN = np.append(datasN,1)
        
    AdjacencyP = coo_matrix((datasP, (rowsP,columnsP)),shape = (n,n)).tocsr()
    AdjacencyP = AdjacencyP + AdjacencyP.transpose()

    AdjacencyN = coo_matrix((datasN, (rowsN,columnsN)),shape = (n,n)).tocsr()
    AdjacencyN = AdjacencyN + AdjacencyN.transpose()

    AdjacencyP.data[:] = 1
    AdjacencyN.data[:] = 1

    return AdjacencyP, AdjacencyN



#Iterative Power Method
def IPM(matrix, niter, n,K ):
    Q,R = np.linalg.qr( np.random.normal(size=(n,K)) )
    for t in range(0,niter):
        Q,R = np.linalg.qr(matrix*Q )
    V = matrix*Q
    return V


#Find Community Indicator Vectors
def genComVec(labels, K,n):
    clusters = []
    vec_clu = np.zeros((K,n))
    clustersize = np.zeros((K), dtype = np.int32)
    for k in range(0,K):
        clusters.append(np.where(labels == k)[0])
        clustersize[k] = len(clusters[k])
        vec_clu[k,clusters[k]] = np.ones(shape=(clustersize[k]), dtype=np.int8)
    return vec_clu

#Greedy Updates
def greedyUpdate(Amatrix, hatp, labels, niter, K, L, n):
    clu_vec = genComVec(labels=labels, K=K, n=n).transpose()
    bhatp = np.log(np.ones(shape=(L,K,K))- hatp )
    lhatp = (np.log(hatp) -bhatp)
    for t in range(0,niter):
        val = np.zeros((n,K))
        for l in range(0,L):
            val += np.ones(shape=(n,1))* np.sum(np.dot(clu_vec,bhatp[l]),axis=0)
            val += Amatrix[l]*np.dot(clu_vec,lhatp[l])
        clu_vec = genComVec(labels=np.argmax(val,axis=1), K=K,n=n).transpose()
    return clu_vec

def greedyUpdateT(AmatrixP,AmatrixN, hatp, labels, niter,n):
    clu_vec = genComVec(labels=labels, K=2, n=n).transpose()
    bhatp = np.log(np.ones(shape=(2,2))- hatp)
    lhatp = np.log(hatp)
    for t in range(0,niter):
        val = np.zeros((n,2))
        val += AmatrixN*np.dot(clu_vec,bhatp)
        val += AmatrixP*np.dot(clu_vec,lhatp)
        clu_vec = genComVec(labels=np.argmax(val,axis=1), K=2,n=n).transpose()
    return clu_vec

def greedyUpdate2T(AmatrixP,AmatrixN,indices1,indices2, hatp, Thr, clu_vec,n):

    rem_vec = np.zeros((n,1))
    bhatp = np.log(np.ones(shape=(2,2))- hatp)
    lhatp = np.log(hatp)

    val = np.zeros((n,2))
    val += AmatrixN*np.dot(clu_vec,bhatp)
    val += AmatrixP*np.dot(clu_vec,lhatp)

    for i in range(0,len(indices1)):
        if val[indices1[i],0] - val[indices1[i],1] > Thr:
            clu_vec[indices1[i],0] = 1
        else:
            clu_vec[indices1[i],0] = 0
            rem_vec[indices1[i]] = 1
    for i in range(0,len(indices2)):
        if val[indices2[i],1] - val[indices2[i],0] > Thr:
            clu_vec[indices2[i],1] = 1
        else:
            clu_vec[indices2[i],1] = 0
            rem_vec[indices2[i]] = 1
            
    return clu_vec, rem_vec

def greedyUpdate3T(AmatrixP,AmatrixN,indices1,indices2,indices3, hatp, clu_vec,n):


    bhatp = np.log(np.ones(shape=(2,2))- hatp)
    lhatp = np.log(hatp)

    val = np.zeros((n,2))
    val += AmatrixN*np.dot(clu_vec,bhatp)
    val += AmatrixP*np.dot(clu_vec,lhatp)

    for i in range(0,len(indices3)):
        if val[indices3[i],0] > val[indices3[i],1] :
            clu_vec[indices3[i],0] = 1
        else:
            clu_vec[indices3[i],1] = 1
            
    return clu_vec


#Estimate Parameters
def getP(Amatrix, labels,L,K):
    hatp = np.zeros((L,K,K),dtype=np.float) 
    clusters = []
    clustersize = np.zeros((K), dtype = np.int32)
    for k in range(0,K):
        clusters.append(np.where(labels == k)[0])
        clustersize[k] = len(clusters[k])
    
    for l in range(0,L):
        for i in range(0,K):
            for j in range(i,K):
                Acounter = float(Amatrix[l][clusters[i],:][:,clusters[j]].getnnz())
                hatp[l,i,j] = (Acounter/clustersize[i])/clustersize[j]
                hatp[l,j,i] = hatp[l,i,j]
    return hatp

def getPT(AmatrixP,AmatrixN, labels):
    hatp = np.zeros((2,2),dtype=np.float) 
    clusters = []
    clustersize = np.zeros((2), dtype = np.int32)
    for k in range(0,2):
        clusters.append(np.where(labels == k)[0])
        clustersize[k] = len(clusters[k])
    
    for i in range(0,2):
        for j in range(i,2):
            AcounterP = float(AmatrixP[clusters[i],:][:,clusters[j]].getnnz())
            AcounterN = float(AmatrixN[clusters[i],:][:,clusters[j]].getnnz())
            hatp[i,j] = (AcounterP/(AcounterP+AcounterN+1))
            hatp[j,i] = hatp[i,j]
    return hatp

def SpectralCluster(matrix, niter, n, K):
    #Iterative Power Method
    V = IPM(matrix= matrix , niter = niter, n=n, K=K )
    
    #Kmeans
    KMpp = KMeans(init='k-means++', n_clusters=K, n_init=niter)
    KMpp.fit(V)

    return KMpp.labels_

def KLdiv(p,q):
    return sum(p*np.log(p/q))

def DLP(p,q):
    y = np.sqrt(p*q)
    return KLdiv(np.array([y,1-y]),np.array([p,1-p])) + KLdiv(np.array([y,1-y]),np.array([q,1-q]))


#h5 file version

#Create Adjacency Matrices and save to h5 file
def genAmatrixH(nk,cnk,p_in,p_out,K,filename):
    n = nk[K-1] + cnk[K-1]
    f = tb.open_file(filename, 'w')
    filters = tb.Filters(complevel=5, complib='blosc')
    out_indices = f.create_earray(f.root, 'indices', tb.Int32Atom(),shape=(0,), filters=filters)
    out_indptr = f.create_carray(f.root, 'indptr', tb.Int32Atom(), shape=(n+1,), filters=filters)
    out_indptr[0] = 0
    
    for k in range(0,K):
        for i in range(0,nk[k]):
            con = rand(1,nk[k]-i-1,density = p_in, format = 'csr')
            con.indices[:] = con.indices[:] + cnk[k]+i+1
            out_indices.append(con.indices)
            out_indptr[i+cnk[k]+1] = out_indptr[i+cnk[k]] + con.getnnz()
            for j in range(k+1,K):
                con = rand(1,nk[j], density= p_out, format = 'csr')
                con.indices[:] = con.indices[:] + cnk[j]
                out_indices.append(con.indices)
                out_indptr[i+cnk[k]+1] += con.getnnz()
    f.close()


#Iterative Power Method for Upper Triangel Matrix
def IPMH(h5, niter, n,K ,bs):
    L = len(h5)
    weight = np.random.rand((L))
    weight[:] -= 0.5
    Q,R = np.linalg.qr( np.random.normal(size=(n,K)) )
    for t in range(0,niter):
        Vtemp = np.zeros((n,K))
        for v in range(0,n,bs):
            imax = min(v+bs, n)
            dv = imax-v
            for l in range(0,L):
                indptr= h5[l].root.indptr[v:imax+1]
                indices = h5[l].root.indices[indptr[0]:indptr[dv]]
                indptr[:] -= indptr[0]
                data = np.ones(shape=(indptr[dv]))
                Vtemp[v:imax,:] += weight[l] * csr_matrix((data,indices,indptr),shape=(dv,n)) * Q
                Vtemp += weight[l] * csc_matrix((data,indices,indptr),shape=(n,dv)) * Q[v:imax,:]
        Q,R = np.linalg.qr( Vtemp )
    V = np.zeros((n,K))
    for v in range(0,n,bs):
        imax = min(v+bs, n)
        dv = imax-v
        for l in range(0,L):
            indptr= h5[l].root.indptr[v:imax+1]
            indices = h5[l].root.indices[indptr[0]:indptr[dv]]
            indptr[:] -= indptr[0]
            data =  np.ones(shape=(indptr[dv]))
            V[v:imax,:] += weight[l]* csr_matrix((data,indices,indptr),shape=(dv,n)) * Q
            V += weight[l]* csc_matrix((data,indices,indptr),shape=(n,dv)) * Q[v:imax,:]
        
    return V

#Greedy Updates for Upper trianglular matrix
def greedyUpdateH(h5, hatp, labels, niter, K, n,bs):
    L = len(h5)
    Q = np.zeros(shape=(L,n,K))
    clu_vec = genComVec(labels=labels, K=K, n=n).transpose()
    bhatp = np.log(np.ones(shape=(L,K,K))- hatp )
    lhatp = (np.log(hatp) -bhatp)
    
    for t in range(0,niter):
        val = np.zeros((n,K))
        for l in range(0,L):
            val += np.ones(shape=(n,1))* np.sum(np.dot(clu_vec,bhatp[l]),axis=0)
            Q[l] = np.dot(clu_vec,lhatp[l])
        for v in range(0,n,bs):
            imax = min(v+bs, n)
            dv = imax-v
            for l in range(0,L):
                indptr= h5[l].root.indptr[v:imax+1]
                indices = h5[l].root.indices[indptr[0]:indptr[dv]]
                indptr[:] -= indptr[0]
                data = np.ones(shape=(indptr[dv]))
                val[v:imax,:] += csr_matrix((data,indices,indptr),shape=(dv,n)) * Q[l]
                val += csc_matrix((data,indices,indptr),shape=(n,dv)) * Q[l,v:imax,:]
        clu_vec = genComVec(labels=np.argmax(val,axis=1), K=K, n=n).transpose()
    return clu_vec


#Estimate Parameters
def getPH(h5, labels,K,n,bs):
    L = len(h5)
    clu_vec = genComVec(labels=labels, K=K, n=n).transpose()
    hatp = np.zeros((L,K,K),dtype=np.float)
    V = np.zeros(shape = (L,n,K))
    for v in range(0,n,bs):
        imax = min(v+bs, n)
        dv = imax-v
        for l in range(0,L):
            indptr= h5[l].root.indptr[v:imax+1]
            indices = h5[l].root.indices[indptr[0]:indptr[dv]]
            indptr[:] -= indptr[0]
            data = np.ones(shape=(indptr[dv]))
            V[l,v:imax,:] += (csr_matrix((data,indices,indptr),shape=(dv,n)) * clu_vec)
            V[l] += csc_matrix((data,indices,indptr),shape=(n,dv)) * clu_vec[v:imax,:]
    for l in range(0,L):
        hatp[l] = np.dot(np.transpose(clu_vec),V[l])
    for i in range(0,K):
        li = len(np.where(labels == i)[0])
        for j in range(0,K):
            lj = len(np.where(labels == j)[0])
            hatp[:,i,j] = (hatp[:,i,j]/li)/lj

    return hatp

def SpectralClusterH(h5, niter, n, K ,bs):
    #Iterative Power Method
    V = IPMH(h5 = h5, niter=niter, n=n,K=K ,bs=bs)
    
    #Kmeans
    KMpp = KMeans(init='k-means++', n_clusters=K, n_init=niter)
    KMpp.fit(V)

    return KMpp.labels_
