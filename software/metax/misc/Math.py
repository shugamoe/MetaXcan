import numpy
from numpy.core import (product, asarray, dot, transpose, multiply, newaxis, maximum)

def _rc(s, tolerance):
    cutoff = tolerance * maximum.reduce(s)
    return cutoff

def _ac(s, tolerance):
    return tolerance

def crpinv(a, rcond=1e-15, epsilon=None):
    """Pseudo inverse, relative cutoff """
    return _inv(a, _rc, rcond, epsilon)

def capinv(a, rcond=1e-15, epsilon=None, group_name=None, sumrule=None):
    """Pseudo inverse, absolute cutoff"""
    return _inv(a, _ac, rcond, epsilon, group_name, sumrule)

def _inv(a, cf, rcond, epsilon, group_name, sumrule):
    """
    modified pseudo inverse
    """

    def _assertNoEmpty2d(*arrays):
        for a in arrays:
            if a.size == 0 and product(a.shape[-2:]) == 0:
                raise RuntimeError("Arrays cannot be empty")

    def _makearray(a):
        new = asarray(a)
        wrap = getattr(a, "__array_prepare__", new.__array_wrap__)
        return new, wrap

    a, wrap = _makearray(a)
    _assertNoEmpty2d(a)

    if epsilon is not None:
        epsilon = numpy.repeat(epsilon, a.shape[0])
        epsilon = numpy.diag(epsilon)
        a = a + epsilon
    a = a.conjugate()
    # WARNING! the "s" eigenvalues might not equal the eigenvalues of eigh
    u, s, vt = numpy.linalg.svd(a, 0)
    m = u.shape[0]
    n = vt.shape[1]
    eigen = numpy.copy(s)
    # numpy.savetxt("intermediate/{}_eigen.txt".format(group_name), eigen) #
    # DEBUG ONLY, remove before commit

    # cutoff = rcond*maximum.reduce(s) 
    if sumrule is not None:
        cutoff = sumrule * numpy.sum(eigen)
        met_sumrule = False
        for i in range(min(n, m)):
            filt=[True if j <= i else False for j in range(len(s))]
            tally = numpy.sum(s[filt])
            if met_sumrule is True:
                s[i] = 0.
            elif (tally >= cutoff):
                met_sumrule = True
        # equivalent to making s[i] = 1. / s[i] for all non-zero s[i]
        s = numpy.concatenate((1 / s[s > 0], s[s == 0])) 
    else:
        cutoff = cf(s, rcond)
        for i in range(min(n, m)):
            # The first Singular Value will always be selected because we want at
            # least one, and the first is the highest
            
            # tally is either s[i] for the normal cutoff condition or sum_{i=0,i=i} s[i] 
            if sumrule is not None:
                filt=[True if j <= i else False for j in range(len(s))]
                tally = numpy.sum(s[filt])
                cutoff = numpy.sum(s) * sumrule # Calculate this each iteration?
            else:
                tally = s[i]
            if tally >= cutoff or i==0:
                s[i] = 1. / s[i]
            else:
                s[i] = 0.

    n_indep = numpy.count_nonzero(s)
    res = dot(transpose(vt), multiply(s[:, newaxis], transpose(u)))
    return wrap(res), n_indep, eigen

def standardize(x):
    mean = numpy.mean(x)
    #follow R's convention, ddof=1
    scale = numpy.std(x, ddof=1)
    if scale == 0:
        return None
    x = x - mean
    x = x / scale
    return x
