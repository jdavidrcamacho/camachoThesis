import numpy as np
import matplotlib
matplotlib.rcParams['figure.figsize'] = [7, 6]
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')

import emcee
from gpyrn import covfunc, meanfunc, meanfield
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.linewidth'] = 2
from matplotlib.ticker import AutoMinorLocator

time,rv,rverr,bis,biserr= np.loadtxt("/home/camacho/Github/camachoThesis/Chapter4Plots/SUN_periodograms/sunBinned_Dumusque.txt", 
                                     skiprows = 1,  unpack = True, usecols = (0,1,2,7,8))
val1, val1err = rv, rverr
val2, val2err = bis, biserr

gprnResults = "/media/camacho/HDD 1 tb/GPRN/01_SUN/70_sun/GPRN/05a_gprn_RVsBIS/savedProgress.h5"
gprnsampler = emcee.backends.HDFBackend(gprnResults)
gprntau = gprnsampler.get_autocorr_time(tol=0)
gprnburnin = int(2*np.max(gprntau))
gprnthin = int(0.1 * np.min(gprntau))
gprnsamples = gprnsampler.get_chain(discard=gprnburnin, flat=True, thin=gprnthin)
log_prob_samples = gprnsampler.get_log_prob(discard=gprnburnin, flat=True, thin=gprnthin)
gprnCombSamples = np.concatenate((gprnsamples, log_prob_samples[:, None]), axis=1)


fig, axs = plt.subplots(nrows=4,ncols=1, sharex=True, 
                        gridspec_kw={'width_ratios': [1],
                                     'height_ratios': [3, 1.75, 3, 1.75]})

axs[0].errorbar(time, val1, val1err, fmt= '.k', alpha=0.5)
axs[0].set_ylabel('RV (m/s)')
axs[0].xaxis.set_minor_locator(AutoMinorLocator(5))
axs[0].yaxis.set_minor_locator(AutoMinorLocator(5))
axs[0].grid(which='major', alpha=0.5)
axs[0].grid(which='minor', alpha=0.2)

axs[2].errorbar(time, val2, val2err, fmt= '.k', alpha=0.5)
axs[2].set_ylabel('BIS (m/s)')
axs[2].xaxis.set_minor_locator(AutoMinorLocator(5))
axs[2].yaxis.set_minor_locator(AutoMinorLocator(5))
axs[2].grid(which='major', alpha=0.5)
axs[2].grid(which='minor', alpha=0.2)


##### GPRN stuff
values = np.where(gprnCombSamples[:,-1] == np.max(gprnCombSamples[:,-1]))
gprnMapSample = gprnCombSamples[values,:].reshape(-1, 15)

nodes = [covfunc.QuasiPeriodic(gprnMapSample[-1,0], gprnMapSample[-1,1], 
                               gprnMapSample[-1,2], gprnMapSample[-1,3])]
weight = [covfunc.SquaredExponential(gprnMapSample[-1,4], gprnMapSample[-1,5]),
          covfunc.SquaredExponential(gprnMapSample[-1,6], gprnMapSample[-1,7])]
means = [meanfunc.Linear(gprnMapSample[-1,8], gprnMapSample[-1,9]),
         meanfunc.Linear(gprnMapSample[-1,10], gprnMapSample[-1,11])]
jitter = [gprnMapSample[-1,12], gprnMapSample[-1,13]]

GPRN = meanfield.inference(1, time, val1, val1err, val2, val2err)
elbo, m, v = GPRN.ELBOcalc(nodes, weight, means, jitter, 
                           iterations = 50000, mu='init', var='init')

tstar = np.linspace(time.min()-10, time.max()+10, 2500)
tstar = np.sort(np.concatenate((tstar, time)))

a, b, c = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v)
bmin, bmax = a-np.sqrt(b), a+np.sqrt(b)

aa, bb, _, cc = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v,
                             separate=True)

values = []
for i, j in enumerate(time):
    posVal = np.where(c == j)
    values.append(int(posVal[0]))

val1Pred, val2Pred = [], []
for i, j in enumerate(values):
    val1Pred.append(a[j,0])
    val2Pred.append(a[j,1])

gprnresiduals1 = val1 - np.array(val1Pred)
gprnresiduals2 = val2 - np.array(val2Pred)

from gpyrn import _utils
rms1 = _utils.wrms(gprnresiduals1, val1err)
rms2 = _utils.wrms(gprnresiduals2, val2err)

axs[0].plot(tstar, a[:,0].T, '-b', alpha=0.75, label='GPRN')
axs[0].fill_between(tstar,  bmax[:,0].T, bmin[:,0].T, color="blue", alpha=0.25)
axs[2].plot(tstar, a[:,1].T, '-b', alpha=0.75, label='GPRN')
axs[2].fill_between(tstar,  bmax[:,1].T, bmin[:,1].T, color="blue", alpha=0.25)

axs[1].axhline(y=0, linestyle='--', color='k')
axs[1].errorbar(time, gprnresiduals1, val1err, fmt='.', color='blue')
axs[1].set_ylabel('Residuals (m/s)')
axs[1].xaxis.set_minor_locator(AutoMinorLocator(5))
axs[1].yaxis.set_minor_locator(AutoMinorLocator(5))
axs[1].grid(which='major', alpha=0.5)
axs[1].grid(which='minor', alpha=0.2)
axs[1].text(0.608, 0.935, 'RMS = {0}m/s'.format(round(rms1, 3)),
            bbox={'facecolor':'whitesmoke', 'alpha':1, 'boxstyle':'round'},
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[1].transAxes, color='black', fontsize=10)

axs[3].axhline(y=0, linestyle='--', color='k')
axs[3].errorbar(time, gprnresiduals2, val2err, fmt='.', color='blue')
axs[3].set_ylabel('Residuals (m/s)')
axs[3].xaxis.set_minor_locator(AutoMinorLocator(5))
axs[3].yaxis.set_minor_locator(AutoMinorLocator(5))
axs[3].grid(which='major', alpha=0.5)
axs[3].grid(which='minor', alpha=0.2)
axs[3].text(0.608, 0.935, 'RMS = {0}m/s'.format(round(rms2, 3)),
            bbox={'facecolor':'whitesmoke', 'alpha':1, 'boxstyle':'round'},
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[3].transAxes, color='black', fontsize=10)
axs[3].set_xlabel('Time (BJD - 2400000)')

plt.tight_layout(pad=0.1, h_pad=0.25, w_pad=0.1)
plt.savefig('RVBISfit_withResidualsPlot.pdf', bbox_inches='tight')
plt.close('all')

################################################################################
from artgpn.arttwo import network
import artgpn as art

gpResults = "/media/camacho/HDD 1 tb/GPRN/01_SUN/70_sun/GP/05b_GP_RVsBIS/savedProgress.h5"
gpsampler = emcee.backends.HDFBackend(gpResults)
gptau = gpsampler.get_autocorr_time(tol=0)
gpburnin = int(2*np.max(gptau))
gpthin = int(0.1 * np.min(gptau))
gpsamples = gpsampler.get_chain(discard=gpburnin, flat=True, thin=gpthin)
log_prob_samples = gpsampler.get_log_prob(discard=gpburnin, flat=True, thin=gpthin)
gpCombSamples = np.concatenate((gpsamples, log_prob_samples[:, None]), axis=1)

##### GP stuff
values = np.where(gpCombSamples[:,-1] == np.max(gpCombSamples[:,-1]))
gpMapSample = gpCombSamples[values,:].reshape(-1, 12)

nodes = [art.node.QuasiPeriodic(gpMapSample[-1,0], gpMapSample[-1,1], gpMapSample[-1,2])]
weights = [art.weight.Constant(gpMapSample[-1,3]**2),
           art.weight.Constant(gpMapSample[-1,4]**2)]
means = [art.mean.Linear(gpMapSample[-1,5], gpMapSample[-1,6]),
         art.mean.Linear(gpMapSample[-1,7], gpMapSample[-1,8])]
jitters = [gpMapSample[-1,9], gpMapSample[-1,10]]
GPnet = network(1, time, val1, val1err, val2, val2err)

tstar = np.linspace(time.min()-10, time.max()+10, 2500)
tstar = np.sort(np.concatenate((tstar, time)))

mu11, std11, cov11 = GPnet.prediction(nodes = nodes, weights = weights, means = means,
                                      jitters = jitters, time = tstar, dataset = 1)
mu22, std22, cov22 = GPnet.prediction(nodes = nodes, weights = weights, means = means,
                                      jitters = jitters, time = tstar, dataset = 2)

fig, axs = plt.subplots(nrows=4,ncols=1, sharex=True, 
                        gridspec_kw={'width_ratios': [1],
                                     'height_ratios': [3, 1.75, 3, 1.75]})

axs[0].errorbar(time, val1, val1err, fmt= '.k', alpha=0.5)
axs[0].set_ylabel('RV (m/s)')
axs[0].xaxis.set_minor_locator(AutoMinorLocator(5))
axs[0].yaxis.set_minor_locator(AutoMinorLocator(5))
axs[0].grid(which='major', alpha=0.5)
axs[0].grid(which='minor', alpha=0.2)

axs[2].errorbar(time, val2, val2err, fmt= '.k', alpha=0.5)
axs[2].set_ylabel('BIS (m/s)')
axs[2].xaxis.set_minor_locator(AutoMinorLocator(5))
axs[2].yaxis.set_minor_locator(AutoMinorLocator(5))
axs[2].grid(which='major', alpha=0.5)
axs[2].grid(which='minor', alpha=0.2)

axs[0].fill_between(tstar,mu11+np.sqrt(std11), mu11-np.sqrt(std11), color="red", alpha=0.25)
axs[0].plot(tstar, mu11, '-r', alpha=0.75)
axs[0].errorbar(time,val1, val1err, fmt = "k.")
axs[0].set_ylabel("RV (m/s)")

axs[2].fill_between(tstar, mu22+np.sqrt(std22), mu22-np.sqrt(std22), color="red", alpha=0.25)
axs[2].plot(tstar, mu22, '-r', alpha=0.75)
axs[2].errorbar(time,val2, val2err, fmt = "k.")
axs[2].set_ylabel("BIS (m/s)")

values = []
for i, j in enumerate(time):
    posVal = np.where(cov11 == j)
    values.append(int(posVal[0]))
val1Pred = []
for i, j in enumerate(values):
    val1Pred.append(mu11[j])
gpresiduals1 = val1 - np.array(val1Pred)

values = []
for i, j in enumerate(time):
    posVal = np.where(cov22 == j)
    values.append(int(posVal[0]))
val2Pred = []
for i, j in enumerate(values):
    val2Pred.append(mu22[j])
gpresiduals2 = val2 - np.array(val2Pred)

rms1 = _utils.wrms(gpresiduals1, val1err)
rms2 = _utils.wrms(gpresiduals2, val2err)

axs[1].axhline(y=0, linestyle='--', color='k')
axs[1].errorbar(time, gpresiduals1, val1err, fmt='.', color='red')
axs[1].set_ylabel('Residuals (m/s)')
axs[1].xaxis.set_minor_locator(AutoMinorLocator(5))
axs[1].yaxis.set_minor_locator(AutoMinorLocator(5))
axs[1].grid(which='major', alpha=0.5)
axs[1].grid(which='minor', alpha=0.2)
axs[1].text(0.608, 0.935, 'RMS = {0}m/s'.format(round(rms1, 3)),
            bbox={'facecolor':'whitesmoke', 'alpha':1, 'boxstyle':'round'},
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[1].transAxes, color='black', fontsize=10)

axs[3].axhline(y=0, linestyle='--', color='k')
axs[3].errorbar(time, gpresiduals2, val2err, fmt='.', color='red')
axs[3].set_ylabel('Residuals (m/s)')
axs[3].xaxis.set_minor_locator(AutoMinorLocator(5))
axs[3].yaxis.set_minor_locator(AutoMinorLocator(5))
axs[3].grid(which='major', alpha=0.5)
axs[3].grid(which='minor', alpha=0.2)
axs[3].text(0.608, 0.935, 'RMS = {0}m/s'.format(round(rms2, 3)),
            bbox={'facecolor':'whitesmoke', 'alpha':1, 'boxstyle':'round'},
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[3].transAxes, color='black', fontsize=10)
axs[3].set_xlabel('Time (BJD - 2400000)')

plt.tight_layout(pad=0.1, h_pad=0.25, w_pad=0.1)
plt.savefig('RVBISfit_withResidualsPlotb.pdf', bbox_inches='tight')
plt.close('all')