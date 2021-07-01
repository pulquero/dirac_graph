"""
Dirac on a Graph API.
Mark Hale

Ref:
 - The Dirac operator of a graph, Oliver Knill (https://arxiv.org/abs/1306.2166)
"""
from itertools import groupby
import numpy as np


def is_sublist(l,ll):
    Nl = len(l)
    for i in range(0, len(ll)-Nl):
        if l==ll[i:i+Nl]:
            return True
    return False


class Cliques(list):
    def __init__(self, total_count):
        self.total_count = total_count

    def dim(self, i):
        return len(self[i])


class Clique:
    def __init__(self, c):
        self.list = c
        # store set for use in inclusion()
        self.set = frozenset(c)

    def inclusion(self, clique):
        if self.set <= clique.set:
            if len(self.list)>1:
                return 1 if is_sublist(self.list, clique.list+clique.list) else -1
            else:
                return 1 if self.list[0]==clique.list[len(clique.list)-1] else -1
        else:
            return 0


def cliques_by_dim(g, max_dim=None):
    """
    Return the cliques/simplices of a graph in a list, indexed by dimension.
    Each list entry is a list of Clique objects of the same dimension.
    """
    cliques = g.as_undirected().cliques(max=max_dim if max_dim is not None else 0)

    def orient_clique(c):
        # orient 1-simplices/cliques according to edge direction (if available)
        # and leave everything else with its default ordering
        if len(c) == 2:
            return (c[0], c[1]) if g.get_eid(c[0], c[1], error=False)==1 else (c[1],c[0])
        else:
            return c

    oriented_cliques = [orient_clique(c) for c in cliques]
    # sort by dimension (and then by IDs)
    N = len(g.vs)
    sorted_cliques = sorted(oriented_cliques, key=lambda x:len(x)+sum(x)/(N*len(x)))
    clqs = Cliques(len(cliques))
    for _,cs in groupby(sorted_cliques, key=len):
        clqs.append([Clique(c) for c in cs])
    return clqs


def dirac(clqs):
    """
    Returns the Dirac matrix.
    """
    N = len(clqs)
    Nd = clqs.total_count
    D = np.zeros((Nd,Nd), np.int8)
    i_offset = 0
    for i,clqs_i in enumerate(clqs):
        i_size = len(clqs_i)
        for j,c in enumerate(clqs_i):
            if i+1 < N:
                for k,d in enumerate(clqs[i+1]):
                    value = c.inclusion(d)
                    if value != 0:
                        D[i_offset+j][i_offset+i_size+k] = value
                        D[i_offset+i_size+k][i_offset+j] = value
        i_offset += i_size
    return D


def exterior_d(D):
    """
    Returns the exterior derivative.
    """
    return np.tril(D)


def adjoint_d(D):
    """
    Returns the adjoint of the exterior derivative.
    """
    return np.triu(D)


def gamma(clqs):
    """
    Returns the Z_2 grading.
    """
    Nd = clqs.total_count
    g = np.zeros((Nd,Nd), np.int8)
    i_offset = 0
    for i,clqs_i in enumerate(clqs):
        i_size = len(clqs_i)
        for j in range(i_size):
            g[i_offset+j][i_offset+j] = pow(-1, i)
        i_offset += i_size
    return g


def subspace(T, i, clqs, j=None):
    """
    If T is a matrix,
    projects out T_ij : \Omega_i -> \Omega_j from T : \Omega -> \Omega
    (j = i by default).
    Else, if T is a vector,
    projects out T_i \in \Omega_i from T \in \Omega.
    """
    offset_i = sum([clqs.dim(k) for k in range(i)])
    offset_j = sum([clqs.dim(k) for k in range(j)]) if j is not None else offset_i
    N = clqs.dim(i)
    M = clqs.dim(j) if j is not None else N
    if len(T.shape) == 2:
        return T[offset_i:offset_i+N, offset_j:offset_j+M]
    elif j is None:
        return T[offset_i:offset_i+N]
    else:
        raise Exception('Unsupported arguments')


def dirac_space(v, i, clqs):
    """
    Embeds \Omega_i into \Omega.
    """
    offset = sum([clqs.dim(j) for j in range(i)])
    N = clqs.dim(i)
    if len(v.shape)==2:
        u = np.zeros((clqs.total_count,clqs.total_count))
        u[offset:offset+N,offset:offset+N] = v
    else:
        u = np.zeros(clqs.total_count)
        u[offset:offset+N] = v
    return u


def cohomology_groups(Ls):
    """
    Returns the cohomology groups.
    """
    def zero_eigenvectors(ev, evec):
        evec_zeros = []
        for i in range(len(ev)):
            if np.allclose(0, ev[i]):
                evec_zeros.append(evec[:,i])
        return evec_zeros

    groups = []
    for L_i in Ls:
        ev_i, evec_i = np.linalg.eigh(L_i)
        evec_zeros_i = zero_eigenvectors(ev_i, evec_i)
        groups.append(evec_zeros_i)
    return groups


def im_ker(d):
    u,s,vh = np.linalg.svd(d, full_matrices=True)
    rcond = np.finfo(s.dtype).eps * max(u.shape[0], vh.shape[1])
    tol = np.amax(s) * rcond
    offset = np.sum(s > tol, dtype=int)
    im = u[:,:offset]
    ker = vh[offset:,:].T.conj()
    return im, ker


def remove_kernel(u, evec_zeros):
    """
    Removes the kernel subspace from the given vector.
    """
    for evec_zero in evec_zeros:
        u -= np.dot(u, evec_zero)*evec_zero
    return u


def get_vertex_values(u, clqs):
    N_0 = clqs.dim(0)
    u_v = np.ndarray(N_0) # restriction of u to vertices
    for i,c in enumerate(clqs[0]):
        u_v[c.list[0]] = u[i]
    return u_v


def get_edge_values(u, clqs, g):
    N_0 = clqs.dim(0)
    N_1 = clqs.dim(1)
    u_e = np.ndarray(N_1) # restriction of u to edges
    offset = N_0 if len(u)==clqs.total_count else 0
    for i,c in enumerate(clqs[1]):
        u_e[g.get_eid(c.list[0], c.list[1])] = u[i+offset]
    return u_e


def get_2form_values(u, clqs):
    N_0 = clqs.dim(0)
    N_1 = clqs.dim(1)
    u_2 = {}
    for i,c in enumerate(clqs[2]):
        u_2[c.list] = u[i+N_0+N_1]
    return u_2
