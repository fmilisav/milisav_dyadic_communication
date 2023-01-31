import bct
import numpy as np
from tqdm import tqdm
from sklearn.utils import check_random_state

def strength_preserving_rand(A, rewiring_iter = 10, nstage = 100, niter = 10000,
                             temp = 1000, frac = 0.5,
                             energy_func = None, energy_type = 'euclidean',
                             connected = None, verbose = False, seed = None):

    """
    Degree- and strength-preserving randomization of
    undirected, weighted adjacency matrix A
    Parameters
    ----------
    A : (N, N) array-like
        Undirected symmetric weighted adjacency matrix
    rewiring_iter : int, optional
        Rewiring parameter (each edge is rewired approximately maxswap times).
        Default = 10.
    nstage : int, optional
        Number of annealing stages. Default = 100.
    niter : int, optional
        Number of iterations per stage. Default = 10000.
    temp : float, optional
        Initial temperature. Default = 1000.
    frac : float, optional
        Fractional decrease in temperature per stage. Default = 0.5.
    energy_type: str, optional
        Energy function to minimize. Can be either:
            'euclidean': Sum of squares between strength sequence vectors
                         of the original network and the randomized network
            'max': The single largest value
                   by which the strength sequences deviate
            'mae': Mean absolute error
            'mse': Mean squared error
            'rmse': Root mean squared error
        Default = 'euclidean'.
    energy_func: callable, optional
        Callable with two positional arguments corresponding to
        two strength sequence numpy arrays that returns an energy value.
        Overwrites “energy_type”.
        See “energy_type” for specifying a predefined energy type instead.
    connected: bool, optional
        Maintain connectedness of randomized network.
        By default, this is inferred from data.
    verbose: bool, optional
        Print status to screen at the end of every stage. Default = False.
    seed: float, optional
        Random seed. Default = None.
    Returns
    -------
    B : (N, N) array-like
        Randomized adjacency matrix
    min_energy : float
        Minimum energy obtained by annealing
    Notes
    -------
    Uses Maslov & Sneppen rewiring model to produce a
    surrogate adjacency matrix, B, with the same size, density, degree sequence,
    and weight distribution as A. The weights are then permuted to optimize the
    match between the strength sequences of A and B using simulated annealing.
    References
    -------
    Misic, B. et al. (2015) Cooperative and Competitive Spreading Dynamics
    on the Human Connectome. Neuron.

    2014-2022
    Richard Betzel, Indiana University
    Filip Milisav, McGill University

    Modification History:
    2014: Original (Richard Betzel)
    2022: Python translation, added connectedness-preservation functionality,
          new predefined energy types, and
          user-provided energy callable functionality (Filip Milisav)
    """

    try:
        A = np.array(A)
    except ValueError as err:
        msg = ('A must be array_like. Received: {}.'.format(type(A)))
        raise TypeError(msg) from err

    rs = check_random_state(seed)

    n = A.shape[0]
    s = np.sum(A, axis = 1) #strengths of A

    if connected is None:
        connected = False if bct.number_of_components(A) > 1 else True

    #Maslov & Sneppen rewiring
    if connected:
        B = bct.randmio_und_connected(A, rewiring_iter, seed = seed)[0]
    else:
        B = bct.randmio_und(A, rewiring_iter, seed = seed)[0]

    u, v = np.triu(B, k = 1).nonzero() #upper triangle indices
    wts = np.triu(B, k = 1)[(u, v)] #upper triangle values
    m = len(wts)
    sb = np.sum(B, axis = 1) #strengths of B

    if energy_func is not None:
        energy = energy_func(s, sb)
    elif energy_type == 'euclidean':
        energy = np.sum((s - sb)**2)
    elif energy_type == 'max':
        energy = np.max(np.abs(s - sb))
    elif energy_type == 'mae':
        energy = np.mean(np.abs(s - sb))
    elif energy_type == 'mse':
        energy = np.mean((s - sb)**2)
    elif energy_type == 'rmse':
        energy = np.sqrt(np.mean((s - sb)**2))
    else:
        msg = ("energy_type must be one of 'euclidean', 'max', "
               "'mae', 'mse', or 'rmse'. Received: {}.".format(energy_type))
        raise ValueError(msg)

    energymin = energy
    wtsmin = wts

    if verbose:
        print('\ninitial energy {:.5f}'.format(energy))

    for istage in tqdm(range(nstage), desc = 'annealing progress'):

        naccept = 0
        for i in range(niter):

            #permutation
            e1 = rs.randint(m)
            e2 = rs.randint(m)

            a, b = u[e1], v[e1]
            c, d = u[e2], v[e2]

            sb_prime = sb.copy()
            sb_prime[[a, b]] = sb_prime[[a, b]] - wts[e1] + wts[e2]
            sb_prime[[c, d]] = sb_prime[[c, d]] + wts[e1] - wts[e2]

            if energy_func is not None:
                energy_prime = energy_func(sb_prime, s)
            elif energy_type == 'euclidean':
                energy_prime = np.sum((sb_prime - s)**2)
            elif energy_type == 'max':
                energy_prime = np.max(np.abs(sb_prime - s))
            elif energy_type == 'mae':
                energy_prime = np.mean(np.abs(sb_prime - s))
            elif energy_type == 'mse':
                energy_prime = np.mean((sb_prime - s)**2)
            elif energy_type == 'rmse':
                energy_prime = np.sqrt(np.mean((sb_prime - s)**2))
            else:
                msg = ("energy_type must be one of 'euclidean', 'max', "
                       "'mae', 'mse', or 'rmse'. "
                       "Received: {}.".format(energy_type))
                raise ValueError(msg)

            #permutation acceptance criterion
            if (energy_prime < energy or
               rs.rand() < np.exp(-(energy_prime - energy)/temp)):
                sb = sb_prime.copy()
                wts[[e1, e2]] = wts[[e2, e1]]
                energy = energy_prime
                if energy < energymin:
                    energymin = energy
                    wtsmin = wts
                naccept = naccept + 1

        #temperature update
        temp = temp*frac
        if verbose:
            print('\nstage {:d}, temp {:.5f}, best energy {:.5f}, '
                  'frac of accepted moves {:.3f}'.format(istage, temp,
                                                         energymin,
                                                         naccept/niter))

    B = np.zeros((n, n))
    B[(u, v)] = wtsmin
    B = B + B.T

    return B, energymin
