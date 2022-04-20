import numpy as np 
import emcee
import matplotlib.pylab as plt
plt.close('all')
from artgpn.arttwo import network
from artgpn import weight, node, mean

data = np.loadtxt("26965_activity.csv",delimiter=',', skiprows=1, usecols=(1,4,5,11,12))
time = data[:,0].T
val1, val1err = data[:,1].T, data[:,2].T
val2, val2err = data[:,3].T, 2*val1err

nodes = [node.QuasiPeriodic(10, 20, 0.5)] 
weights = [weight.Constant(1), weight.Constant(1)]
means = [mean.Linear(0, np.mean(val1)),
         mean.Linear(0, np.mean(val2))]
jitters =[np.std(val1), np.std(val2)]

#Having defined everything we now create the network we called GPnet
GPnet = network(1, time, val1, val1err, val2, val2err)

filename =  "savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)

#autocorrelation
tau = sampler.get_autocorr_time(tol=25)
burnin = int(2*np.max(tau))
thin = int(0.1 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)

from priors import priors
prior = priors()

#priors function
def logpriorFunc(theta):
    if np.array(theta).ndim == 1:
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
    else:
        logprior_list = []
        for i in range(theta.shape[0]):
            n2,n3,n4, w1, w2, s1,off1, s2,off2, j1,j2 = theta[i,:]
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
            logprior_list.append(logprior)
        return np.array(logprior_list)
    return logprior
    
#log-likelihood function
def loglikeFunc(theta):
    if np.array(theta).ndim == 1:
        n2,n3,n4, w1, w2, s1,off1, s2,off2, j1,j2 = theta
        nodes = [node.QuasiPeriodic(n2, n3, n4)]
        weights = [weight.Constant(w1**2), weight.Constant(w2**2)]
        means = [mean.Linear(s1, off1), mean.Linear(s2, off2)]
        jitters = [j1, j2]
        logl = GPnet.log_likelihood(nodes, weights, means, jitters)
    else:
        loglike_list = []
        for i in range(theta.shape[0]):
            n2,n3,n4, w1, w2, s1,off1, s2,off2, j1,j2 = theta[i,:]
            nodes = [node.QuasiPeriodic(n2, n3, n4)]
            weights = [weight.Constant(w1**2), weight.Constant(w2**2)]
            means = [mean.Linear(s1, off1), mean.Linear(s2, off2)]
            jitters = [j1, j2]
            logl = GPnet.log_likelihood(nodes, weights, means, jitters)
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

