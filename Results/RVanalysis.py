import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')

import emcee
from corner import corner

import tedi as ted
import gprn as gprn

###### Data .rdb file #####
time,rv,rverr = np.loadtxt("/home/camacho/Github/camachoThesis/sunData/sunBinned_Dumusque.txt", 
                           skiprows = 1, unpack = True, usecols = (0,1,2))
val1, val1err = rv, rverr

gpResults = "/home/camacho/GPRN/01_SUN/65_sun/GP/01a_GP_RV/savedProgress.h5"
gprnResults = "/home/camacho/GPRN/01_SUN/65_sun/GPRN/01a_gprn_RV/savedProgress.h5"

gplabels = np.array(["$\eta_1$", "$\eta_2$", "$\eta_3$", "$\eta_4$", "slope",  "offset", "s"])
gprnlabels = np.array(["$\eta_1$", "$\eta_2$", "$\eta_3$", "$\eta_4$", "$\eta_{1_1}$", 
                       "$\eta_{2_1}$", "slope", "offset", "jitter"])

gpsampler = emcee.backends.HDFBackend(gpResults)
gptau = gpsampler.get_autocorr_time(tol=25)
gpburnin = int(2*np.max(gptau))
gpthin = int(0.1 * np.min(gptau))
gpsamples = gpsampler.get_chain(discard=gpburnin, flat=True, thin=gpthin)
log_prob_samples = gpsampler.get_log_prob(discard=gpburnin, flat=True, thin=gpthin)
gpCombSamples = np.concatenate((gpsamples, log_prob_samples[:, None]), axis=1)
values = np.where((gpCombSamples[:,-1] > -800.0))
gpCombSamples = gpCombSamples[values,:].reshape(-1, 8)

gprnsampler = emcee.backends.HDFBackend(gprnResults)
gprntau = gprnsampler.get_autocorr_time(tol=25)
gprnburnin = int(2*np.max(gprntau))
gprnthin = int(0.1 * np.min(gprntau))
gprnsamples = gprnsampler.get_chain(discard=gprnburnin, flat=True, thin=gprnthin)
log_prob_samples = gprnsampler.get_log_prob(discard=gprnburnin, flat=True, thin=gprnthin)
gprnCombSamples = np.concatenate((gprnsamples, log_prob_samples[:, None]), axis=1)
values = np.where((gprnCombSamples[:,-1] > -850.0))
gprnCombSamples = gprnCombSamples[values,:].reshape(-1, 10)

# labels = np.array(["$\eta_1$", "$\eta_2$", "$\eta_3$", "$\eta_4$"])
# corner1 = corner(gpCombSamples[:,[0,1,2,3]], labels=labels,color='blue', bins = 50,
#                   quantiles=[0.16, 0.5, 0.84], smooth=True, smooth1d=True, 
#                   plot_density=True, plot_contours=True,
#                   fill_contours=True, plot_datapoints=False)
# a = np.array([gprnCombSamples[:,0]*gprnCombSamples[:,4], gprnCombSamples[:,1], 
#      gprnCombSamples[:,2], gprnCombSamples[:,3]]).T
# corner2 = corner(a, 
#                  labels=labels, color="red", bins = 50,
#                   quantiles=[0.16, 0.5, 0.84], smooth=True, smooth1d=True,
#                   fig=corner1,
#                   plot_density=True, plot_contours=True,
#                   fill_contours=True, plot_datapoints=False)

##### GP stuff
values = np.where(gpCombSamples[:,-1] == np.max(gpCombSamples[:,-1]))
gpMapSample = gpCombSamples[values,:].reshape(-1, 8)

kernel = ted.kernels.QuasiPeriodic(gpMapSample[-1,0], gpMapSample[-1,1], 
                               gpMapSample[-1,2], gpMapSample[-1,3])\
         + ted.kernels.WhiteNoise(gpMapSample[-1,6])
mean = ted.means.Linear(gpMapSample[-1,4], gpMapSample[-1,5])
tedibear = ted.process.GP(kernel, mean, time, val1, val1err)
tstar = np.linspace(time.min()-1, time.max()+1, 10000)
m,s,_ = tedibear.prediction(kernel, mean, tstar)

fig, axs = plt.subplots(nrows=2,ncols=1, sharex=True, 
                        gridspec_kw={'width_ratios': [1],
                                     'height_ratios': [2.25, 1]})
fig.set_size_inches(w=7, h=5)
axs[0].errorbar(time, val1, val1err, fmt= '.k')
axs[0].set_ylabel('RV (m/s)')
#axs[0].set_xlabel('Time (BJD-2400000)')
axs[0].plot(tstar, m, '-r', alpha=0.75, label='GP')
axs[0].fill_between(tstar,  m+s, m-s, color="red", alpha=0.25)

gpvals,_,_ = tedibear. prediction(kernel, mean, time, std=False)
gpresiduals = val1 - gpvals
rms = np.sqrt(np.sum((gpresiduals -np.mean(gpresiduals))**2)/time.size)
axs[1].axhline(y=0, linestyle='--', color='k')
axs[1].plot(time, gpresiduals, '*r', alpha=0.75, label='GP')

##### GPRN stuff
values = np.where(gprnCombSamples[:,-1] == np.max(gprnCombSamples[:,-1]))
gprnMapSample = gprnCombSamples[values,:].reshape(-1, 10)
nodes = [gprn.covFunction.QuasiPeriodic(gprnMapSample[-1,0], gprnMapSample[-1,1], 
                                        gprnMapSample[-1,2], gprnMapSample[-1,3])]
weight = [gprn.covFunction.SquaredExponential(gprnMapSample[-1,4], gprnMapSample[-1,5])]
means = [gprn.meanFunction.Linear(gprnMapSample[-1,6], gprnMapSample[-1,7])]
jitter = [gprnMapSample[-1,8]]
GPRN = gprn.meanField.inference(1, time, val1, val1err)
elbo, vm, vv = GPRN.ELBOcalc(nodes, weight, means, jitter, iterations = 5000,
                            mu='init', var='init')

a, b = GPRN.Prediction(nodes, weight, means, jitter, tstar, vm, vv, variance= True)
bmin, bmax = a-np.sqrt(b), a+np.sqrt(b)

axs[0].plot(tstar, a[0].T, '--b', alpha=0.75, label='GPRN')
axs[0].fill_between(tstar,  bmax[0].T, bmin[0].T, color="blue", alpha=0.25)
axs[0].legend(loc='upper right', facecolor='white', framealpha=1, edgecolor='black')

gprnvals, _ = GPRN.Prediction(nodes, weight, means, jitter, time, vm, vv, variance= True)
gprnresiduals = val1 - gprnvals[0]
rms = np.sqrt(np.sum((gprnresiduals - np.mean(gprnresiduals))**2)/time.size)
axs[1].set_ylabel('Residuals (m/s)')
axs[1].plot(time, gprnresiduals, '.b', alpha=0.75, label='GPRN')
axs[1].set_xlabel('Time (BJD - 2400000)')
axs[1].legend(loc='upper left', facecolor='white', framealpha=1, edgecolor='black')
plt.tight_layout()
plt.savefig('RVfit_withResidualsPlot.pdf', bbox_inches='tight')
plt.close('all')







