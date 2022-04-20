import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')

from gprn.covFunction import SquaredExponential, QuasiPeriodic
from gprn.meanFunction import Linear
from gprn.meanField import inference
from gprn import utils

import emcee

time,rv,rverr= np.loadtxt("sunBinned_Dumusque.txt", skiprows = 1, 
                          unpack = True, usecols = (0,1,2))
time, val1, val1err = time, rv, rverr
GPRN = inference(1, time, val1, val1err)


filename =  "savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)

#autocorrelation
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.1 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
logProbSamples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
combSamples = np.concatenate((samples, logProbSamples[:, None]), axis=1)

#checking the logPost that matters to us
values = np.where(combSamples[:,-1] == np.max(combSamples[:,-1]))
mapSample = combSamples[values,:]
mapSample = mapSample.reshape(-1, 10)
nodes = [QuasiPeriodic(mapSample[-1,0], mapSample[-1,1], mapSample[-1,2],
                       mapSample[-1,3])]
weight = [SquaredExponential(mapSample[-1,4], mapSample[-1,5])]
means = [Linear(mapSample[-1,6], mapSample[-1,7])]
jitter = [mapSample[-1,8]]

elbo, m, v = GPRN.ELBOcalc(nodes, weight, means, jitter, 
                              iterations = 50000, mu='init', var='init')
tstar = np.linspace(time.min()-10, time.max()+10, 5000)
tstar = np.sort(np.concatenate((tstar, time)))

a, b, c = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v, 
                          variance=True)
bmin1, bmax1 = a[0]-np.sqrt(b[0]), a[0]+np.sqrt(b[0])

aa, bb, cc = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v,
                             separate=True)


fig = plt.figure(constrained_layout=True, figsize=(8.5, 5))
axs = fig.subplot_mosaic( [['predictive 1', 'weight 1'],
                           ['predictive 1', 'node'],
                           ['predictive 1', 'mean'],],
                          empty_sentinel="X",)

axs['predictive 1'].set(xlabel='Time (BJD - 2450000)', ylabel='RV (m/s)')
axs['predictive 1'].errorbar(time, val1, val1err, fmt= '.k', alpha=0.5)
axs['predictive 1'].plot(tstar, a[0].T, '-', color='grey')
axs['predictive 1'].fill_between(tstar,  bmax1.T, bmin1.T, color="grey", alpha=0.25)
axs['weight 1'].set(xlabel='', ylabel='Weight (m/s)')
axs['weight 1'].plot(tstar, bb[1].T, '-', color='grey')
axs['weight 1'].tick_params(axis='both', which='both', labelbottom=False)
axs['node'].set(xlabel='', ylabel='Node')
axs['node'].plot(tstar, bb[0].T, '-', color='grey')
axs['node'].tick_params(axis='both', which='both', labelbottom=False)
axs['mean'].set(xlabel='Time (BJD - 2450000)', ylabel='Mean (m/s)')
axs['mean'].plot(tstar, means[0](tstar), '-', color='grey')
axs['mean'].tick_params(axis='both', which='both', labelbottom=True)
plt.tight_layout(pad=0.1, h_pad=0.25, w_pad=0.5)

fig.savefig('RV_fullPlots.pdf', bbox_inches='tight')
#plt.close('all')
