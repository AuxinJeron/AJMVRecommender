from numpy import linalg as la
from scipy import linalg as sla
from sklearn.cluster import KMeans
import numpy as np
import math

def normalized_clutering_ng(S, k):
    print(S)
    rows, cols = S.shape
    D = degree_matrix(S)
    X = la.inv(sla.sqrtm(D))
    W = S
    L = D - W
    L_sym = X * L * X

    # computer the eigenvectors
    eig_val, eig_vec = la.eig(L_sym)
    eig_val = eig_val[:k]
    eig_vec = eig_vec[:k]
    print(eig_val)
    eig_vec = np.transpose(eig_vec)

    # normalize
    U = [0 for i in range(rows)]
    for i in range(rows):
        s = 0
        for j in range(k):
            s += eig_vec[i, j] * eig_vec[i, j]
        U[i] = math.sqrt(s)

    for i in range(rows):
        for j in range(k):
            eig_vec[i, j] = eig_vec[i, j] / U[i]

    kmeans = KMeans(n_clusters=k, random_state=0).fit(eig_vec)
    return kmeans.labels_

def degree_matrix(S):
    rows, cols = S.shape
    D = np.matrix(np.zeros((rows, cols)))
    for i in range(rows):
        D[i, i] = np.sum(S[i])
        D[i, i] -= S[i, i]
    return D


