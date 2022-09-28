import numpy as np
from sklearn.utils import check_random_state

def match_length_degree_distribution(W, D, nbins=10, nswap=1000,
                                     replacement=False, weighted=True,
                                     seed=None):
    """
    Generates degree- and edge length-preserving surrogate connectomes.
    Parameters
    ----------
    W : (N, N) array-like
        weighted or binary symmetric connectivity matrix.
    D : (N, N) array-like
        symmetric distance matrix.
    nbins : int
        number of distance bins (edge length matrix is performed by swapping
        connections in the same bin). Default = 10.
    nswap : int
        total number of edge swaps to perform. Recommended = nnodes * 20
        Default = 1000.
    replacement : bool, optional
        if True all the edges are available for swapping. Default= False
    weighted : bool, optional
        Whether to return weighted rewired connectivity matrix. Default = True
    seed : float, optional
        Random seed. Default = None
    Returns
    -------
    newB : (N, N) array-like
        binary rewired matrix
    newW: (N, N) array-like
        weighted rewired matrix. Returns matrix of zeros if weighted=False.
    nr : int
        number of successful rewires
    Notes
    -----
    Takes a weighted, symmetric connectivity matrix `data` and Euclidean/fiber
    length matrix `distance` and generates a randomized network with:
        1. exactly the same degree sequence
        2. approximately the same edge length distribution
        3. exactly the same edge weight distribution
        4. approximately the same weight-length relationship
    Reference
    ---------
    Betzel, R. F., Bassett, D. S. (2018) Specificity and robustness of
    long-distance connections in weighted, interareal connectomes. PNAS.

    Adapted to Python by Justine Hansen, McGill University, 2021
    """

    rs = check_random_state(seed)
    N = len(W)
    # divide the distances by lengths
    bins = np.linspace(D[D.nonzero()].min(), D[D.nonzero()].max(), nbins + 1)
    bins[-1] += 1
    L = np.zeros((N, N))
    for n in range(nbins):
        i, j = np.where(np.logical_and(bins[n] <= D, D < bins[n + 1]))
        L[i, j] = n + 1

    # binarized connectivity
    B = (W != 0).astype(np.int_)

    # existing edges (only upper triangular cause it's symmetric)
    cn_x, cn_y = np.where(np.triu((B != 0) * B, k=1))

    tries = 0
    nr = 0
    newB = np.copy(B)

    while((len(cn_x) >= 2) & (nr < nswap)):
        # choose randomly the edge to be rewired
        r = rs.randint(len(cn_x))
        n_x, n_y = cn_x[r], cn_y[r]
        tries += 1

        # options to rewire with
        # connected nodes that doesn't involve (n_x, n_y)
        index = (cn_x != n_x) & (cn_y != n_y) & (cn_y != n_x) & (cn_x != n_y)
        if(len(np.where(index)[0]) == 0):
            cn_x = np.delete(cn_x, r)
            cn_y = np.delete(cn_y, r)

        else:
            ops1_x, ops1_y = cn_x[index], cn_y[index]
            # options that will preserve the distances
            # (ops1_x, ops1_y) such that
            # L(n_x,n_y) = L(n_x, ops1_x) & L(ops1_x,ops1_y) = L(n_y, ops1_y)
            index = (L[n_x, n_y] == L[n_x, ops1_x]) & (
                L[ops1_x, ops1_y] == L[n_y, ops1_y])
            if(len(np.where(index)[0]) == 0):
                cn_x = np.delete(cn_x, r)
                cn_y = np.delete(cn_y, r)

            else:
                ops2_x, ops2_y = ops1_x[index], ops1_y[index]
                # options of edges that didn't exist before
                index = [(newB[min(n_x, ops2_x[i])][max(n_x, ops2_x[i])] == 0)
                         & (newB[min(n_y, ops2_y[i])][max(n_y,
                                                          ops2_y[i])] == 0)
                         for i in range(len(ops2_x))]
                if(len(np.where(index)[0]) == 0):
                    cn_x = np.delete(cn_x, r)
                    cn_y = np.delete(cn_y, r)

                else:
                    ops3_x, ops3_y = ops2_x[index], ops2_y[index]

                    # choose randomly one edge from the final options
                    r1 = rs.randint(len(ops3_x))
                    nn_x, nn_y = ops3_x[r1], ops3_y[r1]

                    # Disconnect the existing edges
                    newB[n_x, n_y] = 0
                    newB[nn_x, nn_y] = 0
                    # Connect the new edges
                    newB[min(n_x, nn_x), max(n_x, nn_x)] = 1
                    newB[min(n_y, nn_y), max(n_y, nn_y)] = 1
                    # one successfull rewire!
                    nr += 1

                    # rewire with replacement
                    if replacement:
                        cn_x[r], cn_y[r] = min(n_x, nn_x), max(n_x, nn_x)
                        index = np.where((cn_x == nn_x) & (cn_y == nn_y))[0]
                        cn_x[index], cn_y[index] = min(
                            n_y, nn_y), max(n_y, nn_y)
                    # rewire without replacement
                    else:
                        cn_x = np.delete(cn_x, r)
                        cn_y = np.delete(cn_y, r)
                        index = np.where((cn_x == nn_x) & (cn_y == nn_y))[0]
                        cn_x = np.delete(cn_x, index)
                        cn_y = np.delete(cn_y, index)

    #ignore message
    #if(nr < nswap):
    #    print(f"I didn't finish, out of rewirable edges: {len(cn_x)}")

    i, j = np.triu_indices(N, k=1)
    # Make the connectivity matrix symmetric
    newB[j, i] = newB[i, j]

    # check the number of edges is preserved
    if(len(np.where(B != 0)[0]) != len(np.where(newB != 0)[0])):
        print(
            f"ERROR --- number of edges changed, \
            B:{len(np.where(B!=0)[0])}, newB:{len(np.where(newB!=0)[0])}")
    # check that the degree of the nodes it's the same
    for i in range(N):
        if(np.sum(B[i]) != np.sum(newB[i])):
            print(
                f"ERROR --- node {i} changed k by: \
                {np.sum(B[i]) - np.sum(newB[i])}")

    newW = np.zeros((N, N))
    if(weighted):
        # Reassign the weights
        mask = np.triu(B != 0, k=1)
        inids = D[mask]
        iniws = W[mask]
        inids_index = np.argsort(inids)
        # Weights from the shortest to largest edges
        iniws = iniws[inids_index]
        mask = np.triu(newB != 0, k=1)
        finds = D[mask]
        i, j = np.where(mask)
        # Sort the new edges from the shortest to the largest
        finds_index = np.argsort(finds)
        i_sort = i[finds_index]
        j_sort = j[finds_index]
        # Assign the initial sorted weights
        newW[i_sort, j_sort] = iniws
        # Make it symmetrical
        newW[j_sort, i_sort] = iniws

    return newB, newW, nr
