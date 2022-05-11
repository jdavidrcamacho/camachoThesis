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

from gprn.covFunction import SquaredExponential, QuasiPeriodic
from gprn.meanFunction import Constant
from gprn.meanField import inference
from gprn.utils import wrms
import emcee

time,rv,rverr,bis,biserr,fw, fwerr = np.loadtxt("sample50points.txt",
                                                skiprows = 1, unpack = True, 
                                                usecols = (0,1,2,3,4,5,6))
val1, val1err = rv, rverr
val3,val3err = fw, fwerr
GPRN = inference(1, time, val1,val1err,val3,val3err)



fig = plt.figure(constrained_layout=False, figsize=(4, 4))
axs = fig.subplot_mosaic([['predictive1'],
                          ['predictive3'],],
                         empty_sentinel="X",)

axs['predictive1'].set(xlabel='', ylabel='1st timeseries (m/s)')
axs['predictive1'].errorbar(time, val1, val1err, fmt= '.r', alpha=0.75,
                            markersize=5)
axs['predictive1'].xaxis.set_minor_locator(AutoMinorLocator(5))
axs['predictive1'].yaxis.set_minor_locator(AutoMinorLocator(5))
axs['predictive1'].grid(which='major', alpha=0.5)
axs['predictive1'].grid(which='minor', alpha=0.2)
axs['predictive1'].tick_params(axis='both', which='both', labelbottom=False)


axs['predictive3'].set(xlabel='Time (days)', ylabel='2nd timeseries (m/s)')
axs['predictive3'].errorbar(time, val3, val3err, fmt= '.b', alpha=0.75,
                            markersize=5)
axs['predictive3'].xaxis.set_minor_locator(AutoMinorLocator(5))
axs['predictive3'].yaxis.set_minor_locator(AutoMinorLocator(5))
axs['predictive3'].grid(which='major', alpha=0.5)
axs['predictive3'].grid(which='minor', alpha=0.2)
axs['predictive3'].tick_params(axis='both', which='both', labelbottom=True)
fig.savefig('01b_samplesData.pdf', bbox_inches='tight')
plt.close('all')

filename =  "savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)


#autocorrelation
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.5 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
logProbSamples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
combSamples = np.concatenate((samples, logProbSamples[:, None]), axis=1)

fig = plt.figure(constrained_layout=False, figsize=(8.5, 8.5))
axs = fig.subplot_mosaic([['node', 'X'],
                          ['node', 'predictive1'],
                          ['weight1', 'predictive1'],
                          ['weight1', 'predictive3'],
                          ['weight2', 'predictive3'],
                          ['weight2', 'X'],],
                         empty_sentinel="X",)

t,n,w1,w3,w2 = np.loadtxt("componentsPoints.txt", skiprows = 1, unpack = True, 
                           usecols = (0,1,2,3,4))

axs['node'].plot(t, n, '.', color='black', markersize=2.5, alpha=1)
axs['node'].set(xlabel='', ylabel='Node')
axs['node'].xaxis.set_minor_locator(AutoMinorLocator(5))
axs['node'].yaxis.set_minor_locator(AutoMinorLocator(5))
axs['node'].grid(which='major', alpha=0.5)
axs['node'].grid(which='minor', alpha=0.2)
axs['node'].tick_params(axis='both', which='both', labelbottom=False)
axs['weight1'].plot(t, w1, '.', color='red',markersize=2.5,  alpha=1)
axs['weight1'].set(xlabel='', ylabel='Weight (m/s)')
axs['weight1'].xaxis.set_minor_locator(AutoMinorLocator(5))
axs['weight1'].yaxis.set_minor_locator(AutoMinorLocator(5))
axs['weight1'].grid(which='major', alpha=0.5)
axs['weight1'].grid(which='minor', alpha=0.2)
axs['weight1'].tick_params(axis='both', which='both', labelbottom=False)
axs['weight2'].plot(t, w2, '.', color='blue', markersize=2.5, alpha=1)
axs['weight2'].set(xlabel='', ylabel='Weight (m/s)')
axs['weight2'].xaxis.set_minor_locator(AutoMinorLocator(5))
axs['weight2'].yaxis.set_minor_locator(AutoMinorLocator(5))
axs['weight2'].grid(which='major', alpha=0.5)
axs['weight2'].grid(which='minor', alpha=0.2)
axs['weight2'].tick_params(axis='both', which='both', labelbottom=True)
axs['weight2'].set_xlabel('Time (days)')

axs['predictive1'].set(xlabel='', ylabel='1st timeseries (m/s)')
axs['predictive1'].errorbar(time, val1, val1err, fmt= '.r',markersize=2.5,  alpha=1)
axs['predictive1'].xaxis.set_minor_locator(AutoMinorLocator(5))
axs['predictive1'].yaxis.set_minor_locator(AutoMinorLocator(5))
axs['predictive1'].grid(which='major', alpha=0.5)
axs['predictive1'].grid(which='minor', alpha=0.2)
axs['predictive1'].tick_params(axis='both', which='both', labelbottom=False)


axs['predictive3'].set(xlabel='', ylabel='2nd timeseries (m/s)')
axs['predictive3'].errorbar(time, val3, val3err, fmt= '.b', markersize=2.5, alpha=1)
axs['predictive3'].xaxis.set_minor_locator(AutoMinorLocator(5))
axs['predictive3'].yaxis.set_minor_locator(AutoMinorLocator(5))
axs['predictive3'].grid(which='major', alpha=0.5)
axs['predictive3'].grid(which='minor', alpha=0.2)
axs['predictive3'].tick_params(axis='both', which='both', labelbottom=True)

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

a, b, c,_,_,_ = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v)
bmin, bmax = a-np.sqrt(b), a+np.sqrt(b)

values = []
for i, j in enumerate(time):
    posVal = np.where(c == j)
    values.append(int(posVal[0]))
val1Pred = []
for i, j in enumerate(values):
    val1Pred.append(a[0,j])
residuals1 = val1 - np.array(val1Pred)
# axs['residuals1'].set_ylabel('Residuals')
# axs['residuals1'].errorbar(time, residuals1, val1err, fmt= '.r')
# axs['residuals1'].axhline(y=0, linestyle='--', color='k')
# axs['residuals1'].xaxis.set_minor_locator(AutoMinorLocator(5))
# axs['residuals1'].yaxis.set_minor_locator(AutoMinorLocator(5))
# axs['residuals1'].grid(which='major', alpha=0.5)
# axs['residuals1'].grid(which='minor', alpha=0.2)
# axs['residuals1'].tick_params(axis='both', which='both', labelbottom=True)

values = []
for i, j in enumerate(time):
    posVal = np.where(c == j)
    values.append(int(posVal[0]))
val3Pred = []
for i, j in enumerate(values):
    val3Pred.append(a[1,j])
residuals3 = val3 - np.array(val3Pred)
# axs['residuals3'].set_ylabel('Residuals')
# axs['residuals3'].errorbar(time, residuals3, val3err, fmt= '.b')
# axs['residuals3'].axhline(y=0, linestyle='--', color='k')
# axs['residuals3'].set_xlabel('Time (days)')
# axs['residuals3'].xaxis.set_minor_locator(AutoMinorLocator(5))
# axs['residuals3'].yaxis.set_minor_locator(AutoMinorLocator(5))
# axs['residuals3'].grid(which='major', alpha=0.5)
# axs['residuals3'].grid(which='minor', alpha=0.2)

axs['predictive3'].tick_params(axis='both', which='both', labelbottom=True)
axs['predictive3'].set_xlabel('Time (days)')
rms1 = wrms(residuals1, val1err)
print('RMS1 (m/s):', rms1)
rms3 = wrms(residuals3, val3err)
print('RMS3 (m/s):', rms3)

# axs['residuals1'].text(0.658, 1.1, 'RMS = {0}m/s'.format(round(rms1, 3)),
#                        bbox={'facecolor':'whitesmoke', 'alpha':1, 
#                              'boxstyle':'round'},
#                        verticalalignment='bottom', horizontalalignment='right',
#                        transform=axs['residuals1'].transAxes, color='black', 
#                        fontsize=10)

# axs['residuals3'].text(0.658, 1.1, 'RMS = {0}m/s'.format(round(rms3, 3)),
#                        bbox={'facecolor':'whitesmoke', 'alpha':1, 
#                              'boxstyle':'round'},
#                        verticalalignment='bottom', horizontalalignment='right',
#                        transform=axs['residuals3'].transAxes, color='black', 
#                        fontsize=10)
axs['predictive1'].plot(tstar, a[0,:].T, '-',  color='grey', linewidth=0.2)
axs['predictive1'].fill_between(tstar,  bmax[0,:].T, bmin[0,:].T, color="black", alpha=0.25)
axs['predictive3'].plot(tstar, a[1,:].T, '-',  color='grey', linewidth=0.2)
axs['predictive3'].fill_between(tstar,  bmax[1,:].T, bmin[1,:].T, color="black", alpha=0.25)

aa, bb, tt, dd, nVar, wVar = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v,
                                 separate=True)

axs['node'].plot(tstar, -dd[0].T, '-',  color='grey', linewidth=0.2)
bmin = -dd[0,0,:] - np.sqrt(nVar)
bmax = -dd[0,0,:] + np.sqrt(nVar)
axs['node'].fill_between(tstar, np.squeeze(bmin), np.squeeze(bmax), color="black", alpha=0.25)

axs['weight1'].plot(tstar, -dd[1][0].T, '-',  color='grey', linewidth=0.2)
bmin = np.squeeze(-dd[1][0]) - np.squeeze(np.sqrt(wVar[0,:]))
bmax = np.squeeze(-dd[1][0]) + np.squeeze(np.sqrt(wVar[0,:]))
axs['weight1'].fill_between(tstar, bmin, bmax, color="black", alpha=0.25)

axs['weight2'].plot(tstar, -dd[1][1].T, '-', color='grey',  linewidth=0.2)
bmin = np.squeeze(-dd[1][1]) - np.squeeze(np.sqrt(wVar[1,:]))
bmax = np.squeeze(-dd[1][1]) + np.squeeze(np.sqrt(wVar[1,:]))
axs['weight2'].fill_between(tstar, bmin, bmax, color="black", alpha=0.25)

fig.savefig('allTogether_fullNEW.pdf', bbox_inches='tight')
plt.show()
plt.close('all')

