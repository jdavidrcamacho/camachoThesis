import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
#from gprn import weightFunction, nodeFunction 
#from gprn import, nodeFunction
from gpyrn import meanfield, covfunc, meanfunc
import emcee

time,rv,rverr,bis,biserr,fw, fwerr = np.loadtxt("sample50points.txt",
                                                skiprows = 1, unpack = True, 
                                                usecols = (0,1,2,3,4,5,6))
val1, val1err = rv, rverr
val2,val2err = bis, biserr
val3,val3err = fw, fwerr
GPRN = meanfield.inference(1, time, val1,val1err, val2,val2err, val3,val3err)

filename =  "savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)

#autocorrelation
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.5 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
combSamples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)

##### Setting priors #####
from scipy import stats
stats.loguniform = stats.reciprocal
from priors import priors
prior = priors(filename="sample50points.txt", 
               RVs=True, BIS=True, FWHM=True)

#priors
def logpriorFunc(theta):
    if np.array(theta).ndim == 1:
        n2,n3,n4, w11,w21,w12,w22,w13,w23, j1,j2,j3 =  theta
        logcdfQPeta2 = prior[1].logcdf(0.5*n3)
        logprior = prior[1].logpdf(n2) - logcdfQPeta2
        logprior += prior[2].logpdf(n3)
        logprior += prior[3].logpdf(n4)
        logprior += prior[4].logpdf(w11)
        logprior += prior[5].logpdf(w21)
        logprior += prior[6].logpdf(w12)
        logprior += prior[7].logpdf(w22)
        logprior += prior[8].logpdf(w13)
        logprior += prior[9].logpdf(w23)
        logprior += prior[16].logpdf(j1)
        logprior += prior[17].logpdf(j2)
        logprior += prior[18].logpdf(j3)
    else:
        logprior_list = []
        for i in range(theta.shape[0]):
            n2,n3,n4, w11,w21,w12,w22,w13,w23, j1,j2,j3 = theta[i,:]
            logcdfQPeta2 = prior[1].logcdf(0.5*n3)
            logprior = prior[1].logpdf(n2) - logcdfQPeta2
            logprior += prior[2].logpdf(n3)
            logprior += prior[3].logpdf(n4)
            logprior += prior[4].logpdf(w11)
            logprior += prior[5].logpdf(w21)
            logprior += prior[6].logpdf(w12)
            logprior += prior[7].logpdf(w22)
            logprior += prior[8].logpdf(w13)
            logprior += prior[9].logpdf(w23)
            logprior += prior[16].logpdf(j1)
            logprior += prior[17].logpdf(j2)
            logprior += prior[18].logpdf(j3)
            logprior_list.append(logprior)
        return np.array(logprior_list)
    return logprior 
        
#MCMC ELBO
def loglikeFunc(theta):
    if np.array(theta).ndim == 1:
        n2,n3,n4, w11,w21,w12,w22,w13,w23, j1,j2,j3 = theta
        nodes = [covfunc.QuasiPeriodic(1, n2, n3, n4)]
        weight = [covfunc.SquaredExponential(w11, w21),
                  covfunc.SquaredExponential(w12, w22),
                  covfunc.SquaredExponential(w13, w23)]
        means = [meanfunc.Constant(0), 
                 meanfunc.Constant(0), 
                 meanfunc.Constant(0)]
        jitter = [j1, j2, j3]
        elbo, m, v = GPRN.ELBOcalc(nodes, weight, means, jitter,
                               iterations=5000, mu='init', var='init')
    else:
        loglike_list = []
        for i in range(theta.shape[0]):
            n2,n3,n4, w11,w21,w12,w22,w13,w23, j1,j2,j3 = theta[i,:]
            nodes = [covfunc.QuasiPeriodic(1, n2, n3, n4)]
            weight = [covfunc.SquaredExponential(w11, w21),
                      covfunc.SquaredExponential(w12, w22),
                      covfunc.SquaredExponential(w13, w23)]
            means = [meanfunc.Constant(0), 
                     meanfunc.Constant(0), 
                     meanfunc.Constant(0)]
            jitter = [j1, j2, j3]
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
                                     densityestimation='kde', 
                                     errorestimation=True)
print('Perrakis normal evidence: {0}'.format(evidence), file=f)

f.close()


