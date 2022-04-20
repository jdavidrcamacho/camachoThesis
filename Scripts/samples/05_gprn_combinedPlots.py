import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.linewidth'] = 2
plt.close('all')
from matplotlib.ticker import AutoMinorLocator

from gpyrn.covfunc import SquaredExponential, QuasiPeriodic
from gpyrn.meanfunc import Constant
from gpyrn.meanfield import inference
from gpyrn._utils import wrms
import emcee

time,rv,rverr,bis,biserr,fw, fwerr = np.loadtxt("sample50points.txt",
                                                skiprows = 1, unpack = True, 
                                                usecols = (0,1,2,3,4,5,6))
val1, val1err = rv, rverr
val2,val2err = fw, fwerr
GPRN = inference(1, time, val1,val1err, val2,val2err)

filename =  "savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)

#autocorrelation
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.5 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
logProbSamples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
combSamples = np.concatenate((samples, logProbSamples[:, None]), axis=1)

fig = plt.figure(constrained_layout=True, figsize=(8.5, 8.5))
axs = fig.subplot_mosaic([['node1', 'predictive1'],
                          ['node1', 'predictive1'],
                          ['node1', 'predictive1'],
                          ['weight1', 'residuals1'],
                          ['weight1', 'residuals1'],
                          ['weight1', 'X'],
                          ['node2', 'predictive2'],
                          ['node2', 'predictive2'],
                          ['node2', 'predictive2'],
                          ['weight2', 'residuals2'],
                          ['weight2', 'residuals2'],
                          ['weight2', 'X'],],
                         empty_sentinel="X",)

t, n,w1,w3,w2 = np.loadtxt("componentsPoints.txt",
                     skiprows = 1, unpack = True, usecols = (0,1,2,3,4))
axs['weight1'].plot(t, w1, 'o', color='red', alpha=0.5)
axs['weight1'].set(xlabel='', ylabel='weight (m/s)')
axs['weight1'].xaxis.set_minor_locator(AutoMinorLocator(5))
axs['weight1'].yaxis.set_minor_locator(AutoMinorLocator(5))
axs['weight1'].grid(which='major', alpha=0.5)
axs['weight1'].grid(which='minor', alpha=0.2)
axs['weight1'].tick_params(axis='both', which='both', labelbottom=False)

axs['node1'].plot(t, n, 'o', color='red', alpha=0.5)
axs['node1'].set(xlabel='', ylabel='node')
axs['node1'].xaxis.set_minor_locator(AutoMinorLocator(5))
axs['node1'].yaxis.set_minor_locator(AutoMinorLocator(5))
axs['node1'].grid(which='major', alpha=0.5)
axs['node1'].grid(which='minor', alpha=0.2)
axs['node1'].tick_params(axis='both', which='both', labelbottom=False)

axs['predictive1'].set(xlabel='', ylabel='data (m/s)')
axs['predictive1'].errorbar(time, val1, val1err, fmt= 'or', alpha=0.5)
axs['predictive1'].xaxis.set_minor_locator(AutoMinorLocator(5))
axs['predictive1'].yaxis.set_minor_locator(AutoMinorLocator(5))
axs['predictive1'].grid(which='major', alpha=0.5)
axs['predictive1'].grid(which='minor', alpha=0.2)
axs['predictive1'].tick_params(axis='both', which='both', labelbottom=False)

axs['weight2'].plot(t, w2, 'o', color='blue', alpha=0.5)
axs['weight2'].set(xlabel='', ylabel='weight (m/s)')
axs['weight2'].xaxis.set_minor_locator(AutoMinorLocator(5))
axs['weight2'].yaxis.set_minor_locator(AutoMinorLocator(5))
axs['weight2'].grid(which='major', alpha=0.5)
axs['weight2'].grid(which='minor', alpha=0.2)
axs['weight2'].tick_params(axis='both', which='both', labelbottom=False)

axs['node2'].plot(t, n, 'o', color='blue', alpha=0.5)
axs['node2'].set(xlabel='', ylabel='node')
axs['node2'].xaxis.set_minor_locator(AutoMinorLocator(5))
axs['node2'].yaxis.set_minor_locator(AutoMinorLocator(5))
axs['node2'].grid(which='major', alpha=0.5)
axs['node2'].grid(which='minor', alpha=0.2)
axs['node2'].tick_params(axis='both', which='both', labelbottom=False)

axs['predictive2'].set(xlabel='', ylabel='data (m/s)')
axs['predictive2'].errorbar(time, val2, val2err, fmt= 'ob', alpha=0.5)
axs['predictive2'].xaxis.set_minor_locator(AutoMinorLocator(5))
axs['predictive2'].yaxis.set_minor_locator(AutoMinorLocator(5))
axs['predictive2'].grid(which='major', alpha=0.5)
axs['predictive2'].grid(which='minor', alpha=0.2)
axs['predictive2'].tick_params(axis='both', which='both', labelbottom=False)

#checking the logPost that matters to us
values = np.where(combSamples[:,-1] == np.max(combSamples[:,-1]))
mapSample = combSamples[values,:]
mapSample = mapSample.reshape(-1, 10)
nodes = [QuasiPeriodic(1,mapSample[-1,0], mapSample[-1,1], mapSample[-1,2])]
weight = [SquaredExponential(mapSample[-1,3], mapSample[-1,4]),
          SquaredExponential(mapSample[-1,5], mapSample[-1,6])]
means = [Constant(0), Constant(0)]
jitter = [mapSample[-1,7], mapSample[-1,8]]

elbo, m, v = GPRN.ELBOcalc(nodes, weight, means, jitter, 
                              iterations = 50000, mu='init', var='init')

tstar = np.linspace(time.min()-10, time.max()+10, 5000)
tstar = np.sort(np.concatenate((tstar, time)))

a, b, c = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v)
bmin, bmax = a-np.sqrt(b), a+np.sqrt(b)

values = []
for i, j in enumerate(time):
    posVal = np.where(c == j)
    values.append(int(posVal[0]))
val1Pred = []
for i, j in enumerate(values):
    val1Pred.append(a[j,0])
residuals1 = val1 - np.array(val1Pred)
axs['residuals1'].set_ylabel('Residuals')
axs['residuals1'].errorbar(time, residuals1, val1err, fmt= '.r')
axs['residuals1'].axhline(y=0, linestyle='--', color='k')
axs['residuals1'].set_xlabel('Time (days)')

values = []
for i, j in enumerate(time):
    posVal = np.where(c == j)
    values.append(int(posVal[0]))
val2Pred = []
for i, j in enumerate(values):
    val2Pred.append(a[j,1])
residuals2 = val2 - np.array(val2Pred)
axs['residuals2'].set_ylabel('Residuals')
axs['residuals2'].errorbar(time, residuals2, val2err, fmt= '.b')
axs['residuals2'].axhline(y=0, linestyle='--', color='k')
axs['residuals2'].set_xlabel('Time (days)')



rms1 = wrms(residuals1, val1err)
print('RMS1 (m/s):', rms1)
rms2 = wrms(residuals2, val2err)
print('RMS2 (m/s):', rms2)

aa, bb, dd,nVar,wVar = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v,
                                 separate=True)

axs['residuals1'].text(0.658, 0.935, 'RMS = {0}m/s'.format(round(rms1, 3)),
                       bbox={'facecolor':'whitesmoke', 'alpha':1, 
                             'boxstyle':'round'},
                       verticalalignment='bottom', horizontalalignment='right',
                       transform=axs['residuals1'].transAxes, color='black', 
                       fontsize=10)

axs['residuals2'].text(0.658, 0.935, 'RMS = {0}m/s'.format(round(rms2, 3)),
                       bbox={'facecolor':'whitesmoke', 'alpha':1, 
                             'boxstyle':'round'},
                       verticalalignment='bottom', horizontalalignment='right',
                       transform=axs['residuals2'].transAxes, color='black', 
                       fontsize=10)


axs['predictive1'].plot(tstar, a[:,0].T, '-k')
axs['predictive1'].fill_between(tstar,  bmax[:,0].T, bmin[:,0].T, color="black", alpha=0.25)
axs['predictive2'].plot(tstar, a[:,1].T, '-k')
axs['predictive2'].fill_between(tstar,  bmax[:,1].T, bmin[:,1].T, color="black", alpha=0.25)

axs['weight1'].plot(tstar, -dd[1][0].T, '-k')
axs['node1'].plot(tstar, -dd[0].T, '-k')
axs['weight2'].plot(tstar, -dd[1][1].T, '-k')
axs['node2'].plot(tstar, -dd[0].T, '-k')

fig.savefig('allTogether_full.pdf', bbox_inches='tight')
# plt.close('all')
