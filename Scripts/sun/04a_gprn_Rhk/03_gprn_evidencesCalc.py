import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
#from gprn import weightFunction, nodeFunction 
from gprn.covFunction import SquaredExponential, QuasiPeriodic, Piecewise
from gprn.meanFunction import Linear
from gprn.meanField import inference
import emcee

labels = np.array(["eta1", "eta2", "eta3", "eta4",
                   "eta1", "eta2", "slope1", "offset1", "jitter1", "elbo"])

time,rhk,rhkerr= np.loadtxt("sunBinned_Dumusque.txt", skiprows = 1, unpack = True,
                          usecols = (0,3,4))
time, val1, val1err = time, rhk, rhkerr
GPRN = inference(1, time, val1, val1err)

filename =  "savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)

#autocorrelation
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.1 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
all_samples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)

##### Setting priors #####
from scipy import stats
stats.loguniform = stats.reciprocal
from priors import priors
prior = priors(filename="sunBinned_Dumusque.txt", 
               RVs=False, BIS=False, FWHM=False, RHK=True)

#priors
def logpriorFunc(theta):
    if np.array(theta).ndim == 1:
        n1, n2, n3, n4, w11, w21, s1, off1, j1 = theta
        logcdfQPeta2 = prior[1].logcdf(0.5*n3)
        logcdfSEeta2 = prior[5].logcdf(n2)
        logprior = prior[0].logpdf(n1)
        logprior += prior[1].logpdf(n2)-logcdfQPeta2
        logprior += prior[2].logpdf(n3)
        logprior += prior[3].logpdf(n4)
        logprior += prior[4].logpdf(w11)
        logprior += prior[5].logpdf(w21)-logcdfSEeta2
        logprior += prior[6].logpdf(s1)
        logprior += prior[7].logpdf(off1)
        logprior += prior[8].logpdf(j1)
    else:
        logprior_list = []
        for i in range(theta.shape[0]):
            n1, n2, n3, n4, w11, w21, s1, off1, j1 = theta[i,:]
            logcdfQPeta2 = prior[1].logcdf(0.5*n3)
            logcdfSEeta2 = prior[5].logcdf(n2)
            logprior = prior[0].logpdf(n1)
            logprior += prior[1].logpdf(n2)-logcdfQPeta2
            logprior += prior[2].logpdf(n3)
            logprior += prior[3].logpdf(n4)
            logprior += prior[4].logpdf(w11)
            logprior += prior[5].logpdf(w21)-logcdfSEeta2
            logprior += prior[6].logpdf(s1)
            logprior += prior[7].logpdf(off1)
            logprior += prior[8].logpdf(j1)
            logprior_list.append(logprior)
        return np.array(logprior_list)
    return logprior 
        
#MCMC ELBO
def loglikeFunc(theta):
    if np.array(theta).ndim == 1:
        n1, n2, n3, n4, w11, w21, s1, off1, j1 = theta
        nodes = [QuasiPeriodic(n1, n2, n3, n4)]
        weight = [SquaredExponential(w11, w21)]
        means = [Linear(s1,off1)]
        jitter = [j1]
        elbo, m, v = GPRN.ELBOcalc(nodes, weight, means, jitter,
                               iterations=5000, mu='init', var='init')
    else:
        loglike_list = []
        for i in range(theta.shape[0]):
            print(i, 'of', theta.shape[0])
            n1, n2, n3, n4, w11, w21, s1, off1, j1 = theta[i,:]
            nodes = [QuasiPeriodic(n1, n2, n3, n4)]
            weight = [SquaredExponential(w11, w21)]
            means = [Linear(s1,off1)]
            jitter = [j1]
            elbo, m, v = GPRN.ELBOcalc(nodes, weight, means, jitter,
                               iterations=5000, mu='init', var='init')
            loglike_list.append(elbo)
        return np.array(loglike_list)
    return elbo
    
from gprn.evidenceEstimation import compute_perrakis_estimate
nsamples = 100

storage_name = "10_results.txt"
f = open(storage_name, "a")

print('\nCalculating evidences...\n', file = f)
evidence = compute_perrakis_estimate(samples, loglikeFunc, 
                                     logpriorFunc, nsamples=nsamples,
                                     densityestimation='normal', 
                                     errorestimation=True)
print('Perrakis normal evidence: {0}'.format(evidence), file=f)

f.close()


