#for multiprocessing
import os
os.environ["OMP_NUM_THREADS"] = "1"
ncpu  = 8

#defining iterations
max_n = 100000


import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')
import emcee
from tedi import process, kernels, means
import corner
labels = np.array(["eta1", "eta2", "eta3", "eta4",
                   "slope", "offset", "s", "logP" ])

data = np.loadtxt("101501_activity.csv",delimiter=',', skiprows=1, usecols=(1,4,5,11,12))
time = data[:,0].T
val1, val1err = data[:,1].T, data[:,2].T
val2, val2err = data[:,3].T, 2*val1err

plt.rcParams['figure.figsize'] = [15, 5]
plt.figure()
plt.errorbar(time, val1, val1err, fmt='.k')
plt.title('Data')
plt.xlabel('Time (BJD-2400000)')
plt.ylabel('RV (m/s)')
plt.savefig('01_dataset.png')
plt.close('all')

#because we need to define a "initial" kernel and mean
kernel = kernels.QuasiPeriodic(1, 2, 3, 4)+kernels.WhiteNoise(0.1)
mean = means.Linear(0, -20)
tedibear = process.GP(kernel, mean, time, val1, val1err)

from scipy import stats
stats.loguniform = stats.reciprocal
from priors import priors
prior = priors("101501_activity.csv", RV=True, FWHM=False)

#priors function
def from_prior():
    per = prior[2].rvs()
    newEta2 = stats.loguniform(0.5*per, time.ptp())
    return np.array([prior[0].rvs(), newEta2.rvs(), per, prior[3].rvs(), 
                     prior[4].rvs(), prior[5].rvs(), prior[6].rvs()])

#log-likelihood function
def logpost(theta):
    n1, n2, n3, n4, slp1, off1, s1 = theta
    if n2 < 0.5*n3:
        return -np.inf
    
    logcdfEta2 = prior[1].logcdf(0.5*n3)
    if np.isinf(logcdfEta2):
        return -np.inf

    logprior = prior[0].logpdf(n1)
    logprior += prior[1].logpdf(n2)-logcdfEta2
    logprior += prior[2].logpdf(n3)
    logprior += prior[3].logpdf(n4)
    logprior += prior[4].logpdf(slp1)
    logprior += prior[5].logpdf(off1)
    logprior += prior[6].logpdf(s1)
    if np.isinf(logprior):
        return -np.inf
        
    new_kern = kernels.QuasiPeriodic(n1, n2, n3, n4)\
                +kernels.WhiteNoise(s1)
    new_mean = means.Linear(slp1, off1)
    loglike = tedibear.log_likelihood(new_kern,new_mean, nugget=True)
    if np.isinf(loglike):
        return -np.inf
        
    logPost = logprior + loglike
    return logPost

##### Sampler definition #####
ndim = from_prior().size
nwalkers = 2*ndim

#Set up the backend
filename = "savedProgress.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

from multiprocessing import Pool
pool = Pool(ncpu)
sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, backend=backend, pool=pool)

#Initialize the walkers
p0=[from_prior() for i in range(nwalkers)]
print("\nRunning...")

#We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_n)
#This will be useful to testing convergence
old_tau = np.inf
#Now we'll sample for up to max_n steps
for sample in sampler.sample(p0, iterations=max_n, progress=True):
    # Only check convergence every 100 steps
    if sampler.iteration % 5000:
        continue
    #Compute the autocorrelation time so far
    tau = sampler.get_autocorr_time(tol=0)
    burnin = int(2*np.max(tau))
    thin = int(0.5 * np.min(tau))
    autocorr[index] = np.mean(tau)
    index += 1
    # Check convergence
    converged = np.all(tau * 25 < sampler.iteration)
    #converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        break
    old_tau = tau
    #plotting corner
    samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
    #log_prob_samples = np.nan_to_num(log_prob_samples)
    print("burn-in: {0}".format(burnin))
    print("thin: {0}".format(thin))
    print("flat chain shape: {0}".format(samples.shape))
    all_samples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)
    corner.corner(all_samples[:,:], labels=labels, color="k", bins = 50,
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


labels = np.array(["eta1", "eta2", "eta3", "eta4",
                   "s", "slope", "offset",
                   "log prob"])
corner.corner(sampler.get_chain(flat=True), labels=labels, color="k", bins = 50,
              quantiles=[0.16, 0.5, 0.84], smooth=True, smooth1d=True, 
              show_titles=True, plot_density=True, plot_contours=True,
              fill_contours=True, plot_datapoints=False)
plt.savefig('02_cornerPlot.png')
plt.close('all')

labels = np.array(["eta1", "eta2", "eta3", "eta4",
                   "s", "slope", "offset"])
corner.corner(all_samples[:, :-1], labels=labels, color="k", bins = 50,
              quantiles=[0.16, 0.5, 0.84], smooth=True, smooth1d=True, 
              show_titles=True, plot_density=True, plot_contours=True,
              fill_contours=True, plot_datapoints=False)
plt.savefig('03_cornerPlot.png')
plt.close('all')
