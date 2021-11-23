import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator
plt.close('all')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.linewidth'] = 2
import emcee

###### Data .rdb file #####
time,rv,rverr,rhk,rhkerr,bis,biserr,fw,fwerr = np.loadtxt("/home/camacho/GPRN/Data/sunBinned_Dumusque.txt", 
                                                          skiprows = 1, 
                                                          unpack = True, 
                                                          usecols = (0,1,2,3,4,7,8,9,10))
val1, val1err = rv, rverr
val2, val2err = fw, fwerr
val11err = np.mean(val1err) #* np.ones_like(val1)
val22err = np.mean(val2err) #* np.ones_like(val2)

from correlation_functions import DCF_EK

gpResults = "/media/camacho/HDD 1 tb/GPRN/01_SUN/70_sun/GP/06d_GP_RVsFWHM/savedProgress.h5"
gprnResults = "/media/camacho/HDD 1 tb/GPRN/01_SUN/70_sun/GPRN/06a_gprn_RVsFW/savedProgress.h5"

gpsampler = emcee.backends.HDFBackend(gpResults)
gptau = gpsampler.get_autocorr_time(tol=0)
gpburnin = int(2*np.max(gptau))
gpthin = int(0.1 * np.min(gptau))
gpsamples = gpsampler.get_chain(discard=gpburnin, flat=True, thin=gpthin)
log_prob_samples = gpsampler.get_log_prob(discard=gpburnin, flat=True, thin=gpthin)
gpCombSamples = np.concatenate((gpsamples, log_prob_samples[:, None]), axis=1)

gprnsampler = emcee.backends.HDFBackend(gprnResults)
gprntau = gprnsampler.get_autocorr_time(tol=0)
gprnburnin = int(2*np.max(gprntau))
gprnthin = int(0.1 * np.min(gprntau))
gprnsamples = gprnsampler.get_chain(discard=gprnburnin, flat=True, thin=gprnthin)
log_prob_samples = gprnsampler.get_log_prob(discard=gprnburnin, flat=True, thin=gprnthin)
gprnCombSamples = np.concatenate((gprnsamples, log_prob_samples[:, None]), axis=1)

###################### GP ######################################################
from artgpn.arttwo import network
from artgpn import weight, node, mean

values = np.where(gpCombSamples[:,-1] == np.max(gpCombSamples[:,-1]))
mapSample = gpCombSamples[values,:]
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

val11 = a1 - means[0](tstar)
val22 = a2 - means[1](tstar)


EKbins1 = np.linspace(-30, 30, 61)
C_EK1, C_EK_err1, bins1 = DCF_EK(tstar, val11, val22, val11err, val22err, 
                                 bins=EKbins1)
t_EK1 = 0.5 * (bins1[1:] + bins1[:-1])
m1 = ~np.isnan(C_EK1)

############################ GPRN stuff ########################################
from gprn.covFunction import SquaredExponential, QuasiPeriodic
from gprn.meanFunction import Linear
from gprn.meanField import inference

GPRN = inference(1, time, val1, val1err, val2, val2err)
values = np.where(gprnCombSamples[:,-1] == np.max(gprnCombSamples[:,-1]))
opt_samples = gprnCombSamples[values,:]
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
a, b, c = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v, 
                          variance=True)

val111 = a[0].T - means[0](tstar)
val222 = a[1].T - means[1](tstar)

EKbins2 = np.linspace(-30, 30, 61)
C_EK2, C_EK_err2, bins2 = DCF_EK(tstar, val111, val222, val11err, val22err, 
                                 bins=EKbins2)
t_EK2 = 0.5 * (bins2[1:] + bins2[:-1])
m2 = ~np.isnan(C_EK2)

EKbins3 = np.linspace(-30, 30, 61)
C_EK3, C_EK_err3, bins3 = DCF_EK(time, val1 - np.mean(val1), val2 - np.mean(val2), 
                                 val11err, val22err, 
                                 bins=EKbins3)
t_EK3 = 0.5 * (bins3[1:] + bins3[:-1])
m3 = ~np.isnan(C_EK3)


################################################################################
val11 = a1 - means[0](tstar)
val11 = val11/mapSample[-1,3]
val22 = a2 - means[1](tstar)
val22 = val22/mapSample[-1,4]

val111 = a[0].T - means[0](tstar)
val111 = val111/(opt_samples[-1,0]*opt_samples[-1,4])
val222 = a[1].T - means[1](tstar)
val222 = val222/(opt_samples[-1,0]*opt_samples[-1,6])

fig = plt.figure(constrained_layout=True, figsize=(7, 3))
axs = fig.subplot_mosaic([['GP', '.'],
                          ['GP', 'LAG'],
                          ['GPRN', 'LAG'],
                          ['GPRN', '.'],],
                         gridspec_kw={'width_ratios': [4, 1],
                                      'height_ratios': [1, 10, 10,1]})

axs['GP'].plot(tstar, val11, '-b', label = 'RV', linewidth=1)
axs['GP'].plot(tstar, val22, '--r', label = 'FWHM', linewidth=1)
axs['GP'].set(xlabel='', ylabel='Normalized\nGP predictive')
axs['GP'].tick_params(axis='both', which='both', labelbottom=False)
axs['GP'].legend(loc='upper left', facecolor='whitesmoke', framealpha=1, edgecolor='black')
axs['LAG'].axvline(x=0, linestyle='-', color='gray')

axs['LAG'].xaxis.set_minor_locator(AutoMinorLocator(5))
axs['LAG'].yaxis.set_minor_locator(AutoMinorLocator(5))
axs['LAG'].grid(which='major', alpha=0.5)
axs['LAG'].grid(which='minor', alpha=0.2)
axs['LAG'].plot(t_EK1[m1], C_EK1[m1], '-.k', label = 'GP')
axs['LAG'].set_xlim(-30, 30)
axs['LAG'].set_ylabel('Cross-correlated signal')
axs['LAG'].set_xlabel('Lag')
axs['LAG'].tick_params(axis='both', which='both', labelbottom=True)

axs['GPRN'].plot(tstar, val111, '-b', label = 'RV', linewidth=1)
axs['GPRN'].plot(tstar, val222, '--r', label = 'FWHM', linewidth=1)
axs['GPRN'].set(xlabel='Time (BJD-2400000)', ylabel='Normalized\nGPRN predictive')
axs['GPRN'].tick_params(axis='both', which='both', labelbottom=True)
axs['GPRN'].legend(loc='upper left', facecolor='whitesmoke', framealpha=1, edgecolor='black')

axs['LAG'].set_ylim(-0.3, 1)
axs['LAG'].plot(t_EK2[m2], C_EK2[m2], ':k', label = 'GPRN')
axs['LAG'].set_ylabel('Cross-correlated signal')
axs['LAG'].set_xlabel('Lag (days)')
axs['LAG'].tick_params(axis='both', which='both', labelbottom=True)
axs['LAG'].legend(loc='lower center', bbox_to_anchor=(0., 1, 1, 1),
                  facecolor='whitesmoke', framealpha=1, edgecolor='black')
# axs['LAG'].plot(t_EK3[m3], C_EK3[m3], ':k', label = 'GPRN')
print (t_EK1[m1][np.where((C_EK1[m1] == np.max(C_EK1[m1])))])
axs['GPRN'].xaxis.set_minor_locator(AutoMinorLocator(5))
axs['GPRN'].yaxis.set_minor_locator(AutoMinorLocator(5))
axs['GPRN'].grid(which='major', alpha=0.5)
axs['GPRN'].grid(which='minor', alpha=0.2)
axs['GP'].xaxis.set_minor_locator(AutoMinorLocator(5))
axs['GP'].yaxis.set_minor_locator(AutoMinorLocator(5))
axs['GP'].grid(which='major', alpha=0.5)
axs['GP'].grid(which='minor', alpha=0.2)
plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
fig.savefig('LAG_RVandFWHM.pdf', bbox_inches='tight')
plt.close('all')
