import time
import networkx
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

def compute_alpha(M, delta_f, D, adjacency1, adjacency2, K=0, lamb=1):
    tr_matrix1 = tr_dot(M.T, delta_f)
    tr_matrix2 = tr_dot(delta_f, D.T)
    tr_matrix3 = tr_matrix2
    tr_matrix4 = tr_dot(D.T * adjacency1, D * adjacency2)
    
    if np.max(K) == 0:
        tr_matrix5 = 0
        tr_matrix6 = 0
    else:
        tr_matrix5 = tr_dot(D.T, K)
        tr_matrix6 = tr_dot(M.T, K)

    tr_matrix1m2m3 = tr_matrix1 - tr_matrix2 - tr_matrix3
    alpha_a = tr_matrix1m2m3 + tr_matrix4
    alpha_b = -tr_matrix1m2m3 - tr_matrix1 + lamb * tr_matrix5 - lamb * tr_matrix6
    alpha_op = -alpha_b / (2 * alpha_a)
    return alpha_op

def tr_dot(a, b):
    # Compute np.trace(a*b)
    trace = np.sum(np.multiply(a, b.T))  # 
    return trace

def hugarian(matrix):
    n, m = matrix.shape
    P = np.mat(np.zeros((n, m)))
    row_ind, col_ind = linear_sum_assignment(-matrix)
    P[row_ind, col_ind] = 1
    return P

def graphmatch_ASM(
    adjacency1, adjacency2, K=0, tol=0.1, alpha=1, lamb=1, gamma0=5, adaptive_alpha=0,niter_max=30 
):
    '''
    Find correspondence for pairwise graphs
    
    Input  
    adjacency1/adjacency2:  adjacency matrices for two graphs
    K: product of two feature matrices
    
    Output
    M: Matching matrix
    runtime: Running time
    '''
    starttime = time.perf_counter()
    n, _ = adjacency1.shape
    m, _ = adjacency2.shape
    big_nm = max(n, m)
    N = np.mat(np.ones((n, m))) / n 
    D = np.mat(np.zeros((big_nm, big_nm)))
    gamma = gamma0

    for i in range(niter_max):
        
        N0 = N
        delta_edges = adjacency1 * N * adjacency2
        D[0:n, 0:m] = delta_edges + lamb * K
        gamma = max(gamma -5, gamma0)
        D, gamma = adaptive_softassign(D, gamma)
        
        if adaptive_alpha: # Compute optimal alpha
            alpha = compute_alpha(N0, delta_edges, D[0:n, 0:m], adjacency1, adjacency2, K, lamb)
            if alpha >= 0 and alpha < 1:
                print("Alpha is" + str(alpha_op))
            else:
                alpha = 1
            
            
        N = (1 - alpha) * N + alpha * D[0:n, 0:m]

        err = abs(N0 / N0.max() - N / N.max()).max()
        print(i, err)
        if err < tol:
            print("Converge")
            break
    M = hugarian(N)
    endtime = time.perf_counter()
    runtime = endtime - starttime
    return M, runtime



def softassign(X, gamma=1):
    # Exponentiate the profit matrix X to create J
    n, m = X.shape
    X = X/X.max() 
    beta = np.log((n + m) / 2) * gamma  #
    J = np.exp(beta *(X))
    # Sinkhorn
    S = sinkhorn(J)
    return S


def sinkhorn(M, num_iters=1000,tol=0.05):
    M = np.array(M)
    n, m = M.shape

    # Initialize the scaling factors u and v
    u, v = np.ones(n), np.ones(m) 

    # Run Sinkhorn iterations
    for i in range(num_iters):
        u_new = 1 / np.matmul(M, v)  #Sum of row 
        v_new = 1 / np.matmul(M.T, u_new) #Sum of clo

        if i % 5 ==1: # Stopping test
            res_diff =  np.max(np.abs(u-u_new))
            if res_diff<tol:
                u, v = u_new, v_new 
                break
        
        u, v = u_new, v_new 

    project_matrix = np.outer(u_new,v_new)
    S = np.multiply(project_matrix, M)
    S = np.mat(S)
    return S

def adaptive_softassign(matrix, gamma_ini=1, eps=0.05):
    matrix = np.array(matrix)
    diff = 10
    Ms0 = softassign(matrix, gamma_ini)
    while diff > eps:
        Ms = np.power(Ms0, ((gamma_ini + 1) / gamma_ini))
        Ms = sinkhorn(Ms)
        diff = np.linalg.norm(Ms - Ms0, 1)
        Ms0 = Ms
        gamma_ini += 1
    Ms = np.mat(Ms)
    return Ms, gamma_ini
