import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')

#from gprn import weightFunction, nodeFunction 
from gprn.covFunction import SquaredExponential, QuasiPeriodic
from gprn.meanFunction import Linear
from gprn.meanField import inference
import emcee

labels = np.array(["eta1", "eta2", "eta3", "eta4",
                   "eta11", "eta21","eta12", "eta22",
                   "slope1", "offset1","slope2", "offset2",
                   "jitter1","jitter2", "elbo"])
                   
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
combSamples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)

#checking the logPost that matters to us
values = np.where(combSamples[:,-1] == np.max(combSamples[:,-1]))
mapSample = combSamples[values,:]
mapSample = mapSample.reshape(-1, 15)

nodes = [QuasiPeriodic(mapSample[-1,0], mapSample[-1,1], mapSample[-1,2],
                       mapSample[-1,3])]
weight = [SquaredExponential(mapSample[-1,4], mapSample[-1,5]),
          SquaredExponential(mapSample[-1,6], mapSample[-1,7])]
means = [Linear(mapSample[-1,8], mapSample[-1,9]),
         Linear(mapSample[-1,10], mapSample[-1,11])]
jitter = [mapSample[-1,12], mapSample[-1,13]]

elbo, m, v = GPRN.ELBOcalc(nodes, weight, means, jitter, 
                              iterations = 50000, mu='init', var='init')

tstar = np.linspace(time.min()-10, time.max()+10, 5000)
tstar = np.sort(np.concatenate((tstar, time)))


a, b, c = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v, 
                          variance=True)
bmin1, bmax1 = a[0]-np.sqrt(b[0]), a[0]+np.sqrt(b[0])
bmin2, bmax2 = a[1]-np.sqrt(b[1]), a[1]+np.sqrt(b[1])

aa, bb, cc = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v,
                             separate=True)

values = []
for i, j in enumerate(time):
    posVal = np.where(c == j)
    values.append(int(posVal[0]))

val1Pred, val2Pred = [], []
for i, j in enumerate(values):
    val1Pred.append(a[0][j])
    val2Pred.append(a[1][j])

residuals1 = val1 - np.array(val1Pred)
residuals2 = val2 - np.array(val2Pred)

fig = plt.figure(constrained_layout=True, figsize=(20, 10))
axs = fig.subplot_mosaic( [['predictive 1', 'mean 1'],
                           ['predictive 1', 'weight 1'],
                           ['residuals 1', 'node'],
                           ['predictive 2', 'node'],
                           ['predictive 2', 'weight 2'],
                           ['residuals 2', 'mean 2'],],)

axs['predictive 1'].set(xlabel='', ylabel='RV (m/s)')
axs['predictive 1'].errorbar(time, val1, val1err, fmt= '.k')
axs['predictive 1'].plot(tstar, a[0].T, '-r')
axs['predictive 1'].fill_between(tstar,  bmax1.T, bmin1.T, color="red", alpha=0.25)
axs['residuals 1'].plot(time, residuals1, '.k')
axs['residuals 1'].axhline(y=0, linestyle='--', color='b')
axs['residuals 1'].set(xlabel='BJD - 2450000', ylabel='RV (m/s)')

axs['predictive 2'].set(xlabel='', ylabel='FWHM (m/s)')
axs['predictive 2'].errorbar(time, val2, val2err, fmt= '.k')
axs['predictive 2'].plot(tstar, a[1].T, '-r')
axs['predictive 2'].fill_between(tstar,  bmax2.T, bmin2.T, color="red", alpha=0.25)
axs['residuals 2'].plot(time, residuals2, '.k')
axs['residuals 2'].axhline(y=0, linestyle='--', color='b')
axs['residuals 2'].set(xlabel='BJD - 2450000', ylabel='FWHM (m/s)')

axs['mean 1'].set(xlabel='', ylabel='1st mean')
axs['mean 1'].plot(tstar, means[0](tstar), '-b')
axs['weight 1'].set(xlabel='', ylabel='1st weight')
axs['weight 1'].plot(tstar, bb[1,0].T, '-b')

axs['node'].set(xlabel='', ylabel='Node')
axs['node'].plot(tstar, bb[0,0].T, '-b')

axs['mean 2'].set(xlabel='BJD - 2450000', ylabel='2nd mean')
axs['mean 2'].plot(tstar, means[1](tstar), '-b')
axs['weight 2'].set(xlabel='', ylabel='2nd weight')
axs['weight 2'].plot(tstar, bb[1,1].T, '-b')

fig.savefig('16_fullPlots.pdf', bbox_inches='tight')
plt.close('all')
