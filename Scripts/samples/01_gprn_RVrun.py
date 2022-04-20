#iterations
max_n = 500000

#for multiprocessing
import os
os.environ["OMP_NUM_THREADS"] = "1"
ncpu = 8

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
import emcee
import corner
#from gprn import weightFunction, nodeFunction
from gpyrn import meanfield, covfunc, meanfunc

time,rv,rverr,bis,biserr,fw, fwerr = np.loadtxt("sample50points.txt",
                                                skiprows = 1, unpack = True, 
                                                usecols = (0,1,2,3,4,5,6))
val1, val1err = rv, rverr
val2,val2err = bis, biserr
val3,val3err = fw, fwerr
GPRN = meanfield.inference(1, time, val1,val1err, val3,val3err)

plt.rcParams['figure.figsize'] = [15, 5]
plt.errorbar(time,val1,val1err, fmt='.k', label='data')
plt.legend()
plt.title('Data')
plt.xlabel('Time (BJD-2400000)')
plt.ylabel('RV (m/s)')
plt.savefig('01_dataset.png')
plt.close('all')
print('01_dataset.png done...')

from scipy import stats
stats.loguniform = stats.reciprocal
from priors import priors
prior = priors(filename="sample50points.txt", 
               RVs=True, BIS=True, FWHM=True)

def from_prior():
    new_neta3 = prior[2].rvs()
    new_neta2 = stats.loguniform(0.5*new_neta3, time.ptp()).rvs()
    return np.array([new_neta2, new_neta3, prior[3].rvs(), 
                     prior[4].rvs(), prior[5].rvs(),
                     prior[8].rvs(), prior[9].rvs(),
                     prior[16].rvs(), prior[18].rvs()])

ndim = from_prior().size 
MU = 'init'
VAR = 'init'

def elboCalc(thetas):
    n2,n3,n4, w11,w21,w13,w23, j1,j3 = thetas
    if (n2 < 0.5*n3):
        return -np.inf
        
    logcdfQPeta2 = prior[1].logcdf(0.5*n3)
    if np.isinf(logcdfQPeta2):
        return -np.inf
    
    logprior = prior[1].logpdf(n2) - logcdfQPeta2
    logprior += prior[2].logpdf(n3)
    logprior += prior[3].logpdf(n4)
    logprior += prior[4].logpdf(w11)
    logprior += prior[5].logpdf(w21)
    logprior += prior[8].logpdf(w13)
    logprior += prior[9].logpdf(w23)
    logprior += prior[16].logpdf(j1)
    logprior += prior[18].logpdf(j3)
    if np.isinf(logprior):
        return -np.inf

    nodes = [covfunc.QuasiPeriodic(1, n2, n3, n4)]
    weight = [covfunc.SquaredExponential(w11, w21),
              covfunc.SquaredExponential(w13, w23)]
    means = [meanfunc.Constant(0), 
             meanfunc.Constant(0)]
    jitter = [j1, j3]
    elbo, m, v = GPRN.ELBOcalc(nodes, weight, means, jitter,
                               iterations=5000, mu='init', var='init')
    logpost = logprior + elbo
    return logpost


nwalkers = 2*ndim
#Set up the backend
filename = "savedProgress.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

from multiprocessing import Pool
pool = Pool(ncpu)
sampler = emcee.EnsembleSampler(nwalkers, ndim, elboCalc,
                                backend=backend, pool=pool)
                                
p0=[from_prior() for i in range(nwalkers)]
print("\nRunning...")
#We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_n)
#This will be useful to testing convergence
old_tau = np.inf
#Now we'll sample for up to max_n steps
for sample in sampler.sample(p0, iterations=max_n, progress=True):
    print(sampler.iteration)
    # Only check convergence every 100 steps
    if sampler.iteration % 5000:
        continue
    #Compute the autocorrelation time so far
    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1
    # Check convergence
    converged = np.all(tau * 25 < sampler.iteration)
    #converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        break
    old_tau = tau
    #plotting corner
    tau = sampler.get_autocorr_time(tol=0)
    burnin = int(2*np.max(tau))
    thin = int(0.5 * np.min(tau))
    samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
    print("burn-in: {0}".format(burnin))
    print("thin: {0}".format(thin))
    print("flat chain shape: {0}".format(samples.shape))
    all_samples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)
    corner.corner(all_samples, color="k", bins = 50,
              quantiles=[0.16, 0.5, 0.84], smooth=True, smooth1d=True, 
              show_titles=True, plot_density=True, plot_contours=True,
              fill_contours=True, plot_datapoints=False)
    plt.savefig('tmp_corner_{0}.png'.format(sampler.iteration))
    plt.close('all')


##### Plots #####
#plotting corner
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.5 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
all_samples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)

#plotting corner
import corner

corner.corner(all_samples, color="k", bins = 50,
              quantiles=[0.16, 0.5, 0.84], smooth=True, smooth1d=True, 
              show_titles=True, plot_density=True, plot_contours=True,
              fill_contours=True, plot_datapoints=False)
plt.savefig('02_cornerPlot.png')
plt.close('all')


