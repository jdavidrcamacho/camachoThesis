import numpy as np 
import emcee
import matplotlib.pylab as plt
plt.close('all')
from tedi import process, kernels, means

###### Data .rdb file #####
data = np.loadtxt("101501_activity.csv",delimiter=',', skiprows=1, usecols=(1,4,5,11,12))
time = data[:,0].T
val1, val1err = data[:,1].T, data[:,2].T

kernel = kernels.QuasiPeriodic(1, 2, 3, 4)\
            +kernels.WhiteNoise(0.1)
mean = means.Linear(0, -20)
tedibear = process.GP(kernel, mean, time, val1, val1err)

filename =  "savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)

#autocorrelation
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.1 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)

from priors import priors
prior = priors("101501_activity.csv", RV=True, FWHM=False)

#priors function
def logpriorFunc(theta):
    if np.array(theta).ndim == 1:
        n1, n2, n3, n4, slp1, off1, s1= theta
        logcdfEta2 = prior[1].logcdf(0.5*n3)
        logprior = prior[0].logpdf(n1)
        logprior += prior[1].logpdf(n2)-logcdfEta2
        logprior += prior[2].logpdf(n3)
        logprior += prior[3].logpdf(n4)
        logprior += prior[4].logpdf(slp1)
        logprior += prior[5].logpdf(off1)
        logprior += prior[6].logpdf(s1)
    else:
        logprior_list = []
        for i in range(theta.shape[0]):
            n1, n2, n3, n4, slp1, off1, s1= theta[i,:]
            logcdfEta2 = prior[1].logcdf(0.5*n3)
            logprior = prior[0].logpdf(n1)
            logprior += prior[1].logpdf(n2)-logcdfEta2
            logprior += prior[2].logpdf(n3)
            logprior += prior[3].logpdf(n4)
            logprior += prior[4].logpdf(slp1)
            logprior += prior[5].logpdf(off1)
            logprior += prior[6].logpdf(s1)
            logprior_list.append(logprior)
        return np.array(logprior_list)
    return logprior
    
#log-likelihood function
def loglikeFunc(theta):
    if np.array(theta).ndim == 1:
        n1, n2, n3, n4, slp1, off1, s1= theta
        new_kern = kernels.QuasiPeriodic(n1, n2, n3, n4)\
                    +kernels.WhiteNoise(s1)
        new_mean = means.Linear(slp1, off1)
        logl = tedibear.log_likelihood(new_kern,new_mean)
    else:
        loglike_list = []
        for i in range(theta.shape[0]):
            n1, n2, n3, n4, slp1, off1, s1= theta[i,:]
            new_kern = kernels.QuasiPeriodic(n1, n2, n3, n4)\
                    +kernels.WhiteNoise(s1)
            new_mean = means.Linear(slp1, off1)
            logl = tedibear.log_likelihood(new_kern,new_mean)
            loglike_list.append(logl)
        return np.array(loglike_list)
    return logl

from tedi.evidence import compute_perrakis_estimate
nsamples = 100

storage_name = "10_results.txt"
f = open(storage_name, "a")
print('\nCalculating evidences...\n')
evidence = compute_perrakis_estimate(samples, loglikeFunc, 
                                     logpriorFunc, nsamples=nsamples,
                                     densityestimation='kde', 
                                     errorestimation=True)
print('Perrakis kde evidence: {0}'.format(evidence), file=f)
f.close()

