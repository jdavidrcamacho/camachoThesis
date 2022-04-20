import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
#from gprn import weightFunction, nodeFunction 
from gprn.covFunction import SquaredExponential, QuasiPeriodic, Piecewise
from gprn.meanFunction import Linear
from gprn.meanField import inference
import emcee

labels = np.array(["eta1", "eta2", "eta3", "eta4",
                   "eta11", "eta21", "eta12", "eta22",
                   "slope1", "offset1", "slope2", "offset2", 
                   "jitter1", "jitter2", "elbo"])

data = np.loadtxt("101501_activity.csv",delimiter=',', skiprows=1, usecols=(1,4,5,11,12))
time = data[:,0].T
val1, val1err = data[:,1].T, data[:,2].T
val2, val2err = data[:,3].T, 2*val1err
GPRN = inference(1, time, val1, val1err, val2, val2err)


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
prior = priors()

#priors
def logpriorFunc(theta):
    if np.array(theta).ndim == 1:
        n1,n2,n3,n4, w11,w21, w12,w22, s1,off1, s2,off2, j1,j2 =  theta
        logprior = prior[0].logpdf(n1)
        logprior += prior[1].logpdf(n2)
        logprior += prior[2].logpdf(n3)
        logprior += prior[3].logpdf(n4)
        logprior += prior[4].logpdf(w11)
        logprior += prior[5].logpdf(w21)
        logprior += prior[6].logpdf(w12)
        logprior += prior[7].logpdf(w22)
        logprior += prior[8].logpdf(s1)
        logprior += prior[9].logpdf(off1)
        logprior += prior[10].logpdf(s2)
        logprior += prior[11].logpdf(off2)
        logprior += prior[12].logpdf(j1)
        logprior += prior[13].logpdf(j2)
    else:
        logprior_list = []
        for i in range(theta.shape[0]):
            n1,n2,n3,n4, w11,w21, w12,w22, s1,off1, s2,off2, j1,j2 =  theta[i,:]
            logprior = prior[0].logpdf(n1)
            logprior += prior[1].logpdf(n2)
            logprior += prior[2].logpdf(n3)
            logprior += prior[3].logpdf(n4)
            logprior += prior[4].logpdf(w11)
            logprior += prior[5].logpdf(w21)
            logprior += prior[6].logpdf(w12)
            logprior += prior[7].logpdf(w22)
            logprior += prior[8].logpdf(s1)
            logprior += prior[9].logpdf(off1)
            logprior += prior[10].logpdf(s2)
            logprior += prior[11].logpdf(off2)
            logprior += prior[12].logpdf(j1)
            logprior += prior[13].logpdf(j2)
            logprior_list.append(logprior)
        return np.array(logprior_list)
    return logprior 
        
#MCMC ELBO
def loglikeFunc(theta):
    if np.array(theta).ndim == 1:
        n1,n2,n3,n4, w11,w21, w12,w22, s1,off1, s2,off2, j1,j2 =  theta
        nodes = [QuasiPeriodic(n1, n2, n3, n4)]
        weight = [SquaredExponential(w11, w21), SquaredExponential(w12, w22)]
        means = [Linear(s1, off1), Linear(s2, off2)]
        jitter = [j1, j2]
        elbo, m, v = GPRN.ELBOcalc(nodes, weight, means, jitter,
                               iterations=5000, mu='init', var='init')
    else:
        loglike_list = []
        for i in range(theta.shape[0]):
            n1,n2,n3,n4, w11,w21, w12,w22, s1,off1, s2,off2, j1,j2 =  theta[i,:]
            nodes = [QuasiPeriodic(n1, n2, n3, n4)]
            weight = [SquaredExponential(w11, w21), SquaredExponential(w12, w22)]
            means = [Linear(s1, off1), Linear(s2, off2)]
            jitter = [j1, j2]
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
print('Perrakis kde evidence: {0}'.format(evidence), file=f)

f.close()


