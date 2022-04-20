#for multiprocessing
import os
os.environ["OMP_NUM_THREADS"] = "1"
ncpu  = 8

import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 5]
plt.close('all')

import corner
labels = np.array(["eta3", "eta4", 
                   "weight11", "weight21", "weight11", "weight21",
                   "slope1", "offset1", "slope2", "offset2", 
                   "jitter1", "jitter2", "logl"])
import emcee
max_n = 100000 #defining iterations

from artgpn.arttwo import network
from artgpn import weight, node, mean

data = np.loadtxt("101501_activity.csv",delimiter=',', skiprows=1, usecols=(1,4,5,11,12))
time = data[:,0].T
val1, val1err = data[:,1].T, data[:,2].T
val2, val2err = data[:,3].T, 2*val1err

plt.rcParams['figure.figsize'] = [15, 2*5]
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
axs[0].errorbar(time, val1, val1err, fmt='.k')
axs[0].set_ylabel('RV (m/s)')
axs[1].errorbar(time, val2, val2err, fmt='.k')
axs[1].set_xlabel('Time (BJD-2400000)')
axs[1].set_ylabel('FWHM (m/s)')
plt.savefig('10_dataset.png')
plt.close('all')

nodes = [node.QuasiPeriodic(10, 20, 0.5)] 
weights = [weight.Constant(1), weight.Constant(1)]
means = [mean.Linear(0, np.mean(val1)),
         mean.Linear(0, np.mean(val2))]
jitters =[np.std(val1), np.std(val2)]

#Having defined everything we now create the network we called GPnet
GPnet = network(1, time, val1, val1err, val2, val2err)

loglike = GPnet.log_likelihood(nodes, weights, means, jitters)
print(loglike)


from scipy import stats
stats.loguniform = stats.reciprocal
from priors import priors
prior = priors()

def from_prior():
    return np.array([prior[1].rvs(), prior[2].rvs(), prior[3].rvs(), 
                     prior[4].rvs(), prior[6].rvs(), 
                     prior[8].rvs(), prior[9].rvs(),prior[10].rvs(), prior[11].rvs(),
                     prior[12].rvs(), prior[13].rvs()])


#log_transform calculates our posterior
def log_transform(theta):
    n2,n3,n4, w1, w2, s1,off1, s2,off2, j1,j2 = theta
        
    logprior = prior[1].logpdf(n2)
    logprior += prior[2].logpdf(n3) 
    logprior += prior[3].logpdf(n4)
    logprior += prior[4].logpdf(w1)
    logprior += prior[6].logpdf(w2)
    logprior += prior[8].logpdf(s1)
    logprior += prior[9].logpdf(off1)
    logprior += prior[10].logpdf(s2)
    logprior += prior[11].logpdf(off2)
    logprior += prior[12].logpdf(j1)
    logprior += prior[13].logpdf(j2)
    if np.isinf(logprior):
        return -np.inf

    nodes = [node.QuasiPeriodic(n2, n3, n4)]
    weights = [weight.Constant(w1**2), weight.Constant(w2**2)]
    means = [mean.Linear(s1, off1), mean.Linear(s2, off2)]
    jitters = [j1, j2]

    logpost = logprior + GPnet.log_likelihood(nodes, weights, means, jitters)
    return logpost
##### Sampler definition #####
ndim = from_prior().size
nwalkers = 2*ndim

#Set up the backend
filename = "savedProgress.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

from multiprocessing import Pool
pool = Pool(ncpu)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_transform, 
                                backend=backend, pool=pool)

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
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.5 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
all_samples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)

labels = np.array(["eta2", "eta3", "eta4", 
                   "weight2", "weight1",
                   "slope1", "offset1", "slope2", "offset2", 
                   "jitter1", "jitter2"])
corner.corner(all_samples[:, :-1], labels=labels, color="k", bins = 50,
              quantiles=[0.16, 0.5, 0.84], smooth=True, smooth1d=True, 
              show_titles=True, plot_density=True, plot_contours=True,
              fill_contours=True, plot_datapoints=False)
plt.savefig('02_cornerPlot.png')
plt.close('all')

labels = np.array(["eta2", "eta3", "eta4", 
                   "weight2", "weight1",
                   "slope1", "offset1", "slope2", "offset2", 
                   "jitter1", "jitter2",
                   "log prob"])
corner.corner(sampler.get_chain(flat=True), labels=labels, color="k", bins = 50,
              quantiles=[0.16, 0.5, 0.84], smooth=True, smooth1d=True, 
              show_titles=True, plot_density=True, plot_contours=True,
              fill_contours=True, plot_datapoints=True)
plt.savefig('03_cornerPlot.png')
plt.close('all')

