import numpy as np
import emcee
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')

from artgpn.arttwo import network
from artgpn import weight, node, mean
plt.rcParams['figure.figsize'] = [15, 10]

data = np.loadtxt("10700_activity.csv",delimiter=',', skiprows=1, usecols=(1,4,5,11,12))
time = data[:,0].T
val1, val1err = data[:,1].T, data[:,2].T
val2, val2err = data[:,3].T, 2*val1err

filename =  "/home/camacho/GPRN/02_EXPRES/New/HD10700/02c_GP_RVsFWHM/savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)

#autocorrelation
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.1 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
combSamples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)
values = np.where(combSamples[:,-1] == np.max(combSamples[:,-1]))
mapSample = combSamples[values,:]
mapSample = mapSample.reshape(-1, 12)
nodes = [node.QuasiPeriodic(mapSample[-1,0], mapSample[-1,1], mapSample[-1,2])]
weights = [weight.Constant(mapSample[-1,3]**2),
           weight.Constant(mapSample[-1,4]**2)]
means = [mean.Linear(mapSample[-1,5], mapSample[-1,6]),
         mean.Linear(mapSample[-1,7], mapSample[-1,8])]
jitters = [mapSample[-1,9], mapSample[-1,10]]

GPnet = network(1, time, val1, val1err, val2, val2err)

tstar = np.linspace(time.min()-10, time.max()+10, 5000)
tstar = np.sort(np.concatenate((tstar, time)))

a1, b1, c1 = GPnet.prediction(nodes = nodes, weights = weights, means = means,
                              jitters = jitters,time = tstar, dataset = 1)
bmin1, bmax1 = a1 - b1, a1 + b1
a2, b2, c2 = GPnet.prediction(nodes = nodes, weights = weights, means = means,
                              jitters = jitters,time = tstar, dataset = 2)
bmin2, bmax2 = a2 - b2, a2 + b2

from scipy import signal
rvs = a1 - means[0](tstar)
bis = a2 - means[1](tstar)

corr = signal.correlate(rvs, bis, mode='full', method='auto')
lags = signal.correlation_lags(len(rvs), len(bis), mode='full')
corr /= np.max(corr)

plt.rcParams['figure.figsize'] = [15, 10]
fig, axs = plt.subplots(2, 1, sharex=False, sharey=False)
axs[0].plot(tstar, rvs, '-r', label = 'GP - RV (m/s)')
axs[0].plot(tstar, bis, '-b', label = 'GP - FWHM (m/s)')
axs[0].set_ylabel('m/s')
axs[0].set_xlabel('Time (BJD-2400000)')
axs[0].tick_params(axis='both', which='both', labelbottom=True)
axs[1].plot(lags, corr, '-k', label = 'Lag GP')
axs[1].axvline(x=0, linestyle=':', color='gray')
axs[1].set_xlim(-50, 50)
axs[1].set_ylabel('Cross-correlated signal')
axs[1].set_xlabel('Lag')
axs[1].tick_params(axis='both', which='both', labelbottom=True)
axs[0].legend()
axs[1].legend()


from gprn.covFunction import SquaredExponential, QuasiPeriodic
from gprn.meanFunction import Linear
from gprn.meanField import inference

GPRN = inference(1, time, val1, val1err, val2, val2err)

filename =  "/home/camacho/GPRN/02_EXPRES/New/HD10700/01a_GPRN_RVsFW/savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)

#autocorrelation
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.1 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
all_samples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)
#checking the likelihood that matters to us
values = np.where(all_samples[:,-1] == np.max(all_samples[:,-1]))
opt_samples = all_samples[values,:]
opt_samples = opt_samples.reshape(-1, 15)
nodes = [QuasiPeriodic(opt_samples[-1,0], opt_samples[-1,1], opt_samples[-1,2],
                       opt_samples[-1,3])]
weight = [SquaredExponential(opt_samples[-1,4], opt_samples[-1,5]),
          SquaredExponential(opt_samples[-1,6], opt_samples[-1,7])]
means = [Linear(opt_samples[-1,8], opt_samples[-1,9]),
         Linear(opt_samples[-1,10], opt_samples[-1,11])]
jitter = [opt_samples[-1,12], opt_samples[-1,13]]
elbo, m, v = GPRN.ELBOcalc(nodes, weight, means, jitter, 
                              iterations = 50000, mu='init', var='init')
tstar = np.linspace(time.min()-10, time.max()+10, 5000)
tstar = np.sort(np.concatenate((tstar, time)))
a, b, c = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v, 
                          variance=True)

rvs = a[0].T - means[0](tstar)
bis = a[1].T - means[1](tstar)
corr = signal.correlate(rvs, bis, mode='full', method='auto')
lags = signal.correlation_lags(len(rvs), len(bis), mode='full')
corr /= np.max(corr)

axs[0].plot(tstar, rvs, '--r', label = 'GPRN - RV (m/s)')
axs[0].plot(tstar, bis, '--b', label = 'GPRN - FWHM (m/s)')
axs[1].plot(lags, corr, '--k', label = 'Lag GPRN')
axs[1].tick_params(axis='both', which='both', labelbottom=True)
axs[0].legend()
axs[1].legend()
plt.savefig('17_gp_correlation.pdf', bbox_inches='tight')
plt.close('all')

