import numpy as np
from bct import distance_wei_floyd, retrieve_shortest_path
from tqdm import tqdm
import itertools

def path_transitivity(W, transform = None):

    """
    Computes the path transitivity, or density of one-step detours (triangles)
    available along the shortest path between all pairs of nodes.
    Parameters
    ----------
    W : (N, N) array-like
        Unweighted/Weighted undirected connection weight/length matrix
    transform: callable, optional
        Transform that maps input connection weights to connection lengths
        (only specify if W is a connection weight matrix). Can be either:
            'log': Negative natural logarithm.
            'inv': Inverse.
        Default = None.
    Returns
    -------
    T : (N, N) array-like
        Matrix of pairwise path transitivity
    Notes
    -------
    Translated from the MATLAB function path_transitivity.m,
    openly available in the Brain Connectivity Toolbox
    (https://sites.google.com/site/bctnet)
    Originally written by
    Olaf Sporns, Andrea Avena-Koenigsberger and Joaquin Goñi,
    IU Bloomington, 2014
    References
    -------
    Goñi, J. et al. (2013) Resting-brain functional connectivity predicted by
    analytic measures of network communication. PNAS.
    """

    W = np.array(W)

    n = len(W)
    m = np.zeros((n, n))
    T = np.zeros((n, n))

    for i in range(n - 1):
        for j in range(i + 1, n):
            x = 0
            y = 0
            z = 0
            for k in range(n):
                if W[i, k] != 0 and W[j, k] != 0 and k != i and k != j:
                    x = x + W[i, k] + W[j, k]
                if k != j:
                    y = y + W[i, k]
                if k != i:
                    z = z + W[j, k]
            m[i, j] = x/(y + z)
    m = m + m.T

    hops, Pmat = distance_wei_floyd(W, transform)[1:]

    # --- path transitivity ---
    for i in range(n - 1):
        for j in range(i + 1, n):
            x = 0
            path = retrieve_shortest_path(i, j, hops, Pmat)
            K = len(path)

            for t in range(K - 1):
                for l in range(t + 1, K):
                    x = x + m[path[t], path[l]]

            T[i, j] = 2*x/(K*(K-1))
    T = T + T.T

    return T
