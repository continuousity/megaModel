import numpy as np
from scipy import special
from scipy import optimize

def sad_logLikZ(x):
    """sad_logLikZ -- this takes a 1D array of abundances, fits Fisher's logseries
       to it, then calculates the squared scaled difference between the likelihood
       of the observed data given the model and the likelihood of an ideal dataset"""
    x = x[x > 0]
    
    thisFit = fitFisher(x)
    
    x0 = np.array(range(np.power(10, 6))) + 1
    p0 = np.log(fisherPMF(x0, thisFit[0]))
    p0 = p0[np.isfinite(p0)]
    
    n = x.size
    m = sum(p0 * np.exp(p0)) * n
    v = sum(np.power(m/n - p0, 2) * np.exp(p0)) * n
    
    return np.power((thisFit[1] - m) / np.power(v, 0.5), 2)

def betax(z, a, b):
    """betax -- a helper function for fisherCMF that uses hypergeometric for 
       accuracy and speed"""
    return 1/a * np.power(z, a) * np.power(1-z, b) * special.hyp2f1(a+b, 1, a+1, z)

def fisherCMF(x, beta):
    """fisherCMF -- the CMF for the fisher logseries"""
    return 1 + betax(np.exp(-beta), x+1, 0) / np.log(1 - np.exp(-beta))

def fisherPMF(x, beta):
    """fisherPMF -- the PMF for the fisher logseries"""
    return 1 / np.log(1 / (1 - np.exp(-beta))) * np.exp(-beta * x) / x

def fitFisher(x):
    """ML for fisher logseries"""
    xbar = np.mean(x)
    
    def fun(b): 
        return -1 / np.log(1 - np.exp(-b)) * np.exp(-b)/(1 - np.exp(-b)) - xbar
    
    up = 1.1 * np.exp(2 * np.power(xbar - 0.1, -6)) * np.power(xbar - 0.1, -1.25)
    lo = 0.5 * np.exp(2 * np.power(xbar + 0.1, -5)) * np.power(xbar + 0.1, -1.5)
    
    mle = optimize.brentq(fun, lo, up)
    ll = sum(np.log(fisherPMF(x, mle)))
    
    return [mle, ll]


## example of how it works
# sad_logLikZ(np.array([1, 2, 3, 4]))
