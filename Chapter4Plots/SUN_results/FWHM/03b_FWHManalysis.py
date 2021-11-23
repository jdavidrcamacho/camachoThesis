import numpy as np
import matplotlib
matplotlib.rcParams['figure.figsize'] = [7, 3]
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')


###### Data .rdb file #####
time,rv,rverr = np.loadtxt("/home/camacho/Github/camachoThesis/Chapter4Plots/SUN_periodograms/sunBinned_Dumusque.txt", 
                           skiprows = 1, unpack = True, usecols = (0,9,10))
val1, val1err = rv, rverr

gpResults = "/media/camacho/HDD 1 tb/GPRN/01_SUN/70_sun/GP/03a_GP_FWHM/savedProgress.h5"
gprnResults = "/media/camacho/HDD 1 tb/GPRN/01_SUN/70_sun/GPRN/03a_gprn_FWHM/savedProgress.h5"

gplabels = np.array(["$\eta_1$", "$\eta_2$", "$\eta_3$", "$\eta_4$", "slope",  "offset", "s"])
gprnlabels = np.array(["$\eta_1$", "$\eta_2$", "$\eta_3$", "$\eta_4$", "$\eta_{1_1}$", 
                       "$\eta_{2_1}$", "slope", "offset", "jitter"])

gplabels = np.array(["$\eta_1$", "$\eta_2$", "$\eta_3$", "$\eta_4$", "slope",  "offset", "s"])
gprnlabels = np.array(["$\eta_1$", "$\eta_2$", "$\eta_3$", "$\eta_4$", "$\eta_{1_1}$", 
                       "$\eta_{2_1}$", "slope", "offset", "jitter"])

from gpyrn import covfunc, meanfunc, meanfield
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['figure.figsize'] = [7, 6]

from matplotlib.ticker import AutoMinorLocator
import emcee
import tedi as ted
gpsampler = emcee.backends.HDFBackend(gpResults)
gptau = gpsampler.get_autocorr_time(tol=25)
gpburnin = int(2*np.max(gptau))
gpthin = int(0.1 * np.min(gptau))
gpsamples = gpsampler.get_chain(discard=gpburnin, flat=True, thin=gpthin)
log_prob_samples = gpsampler.get_log_prob(discard=gpburnin, flat=True, thin=gpthin)
gpCombSamples = np.concatenate((gpsamples, log_prob_samples[:, None]), axis=1)

gprnsampler = emcee.backends.HDFBackend(gprnResults)
gprntau = gprnsampler.get_autocorr_time(tol=25)
gprnburnin = int(2*np.max(gprntau))
gprnthin = int(0.1 * np.min(gprntau))
gprnsamples = gprnsampler.get_chain(discard=gprnburnin, flat=True, thin=gprnthin)
log_prob_samples = gprnsampler.get_log_prob(discard=gprnburnin, flat=True, thin=gprnthin)
gprnCombSamples = np.concatenate((gprnsamples, log_prob_samples[:, None]), axis=1)

fig, axs = plt.subplots(nrows=4,ncols=1, sharex=True, 
                        gridspec_kw={'width_ratios': [1],
                                     'height_ratios': [3, 1.75, 3, 1.75]})

axs[0].errorbar(time, val1, val1err, fmt= '.k', alpha=0.5)
axs[0].set_ylabel('FWHM (m/s)')
axs[0].xaxis.set_minor_locator(AutoMinorLocator(5))
axs[0].yaxis.set_minor_locator(AutoMinorLocator(5))
axs[0].grid(which='major', alpha=0.5)
axs[0].grid(which='minor', alpha=0.2)

axs[2].errorbar(time, val1, val1err, fmt= '.k', alpha=0.5)
axs[2].set_ylabel('FWHM (m/s)')
axs[2].xaxis.set_minor_locator(AutoMinorLocator(5))
axs[2].yaxis.set_minor_locator(AutoMinorLocator(5))
axs[2].grid(which='major', alpha=0.5)
axs[2].grid(which='minor', alpha=0.2)

axs[1].xaxis.set_minor_locator(AutoMinorLocator(5))
axs[1].yaxis.set_minor_locator(AutoMinorLocator(5))
axs[1].grid(which='major', alpha=0.5)
axs[1].grid(which='minor', alpha=0.2)

axs[3].xaxis.set_minor_locator(AutoMinorLocator(5))
axs[3].yaxis.set_minor_locator(AutoMinorLocator(5))
axs[3].grid(which='major', alpha=0.5)
axs[3].grid(which='minor', alpha=0.2)

##### GPRN stuff
values = np.where(gprnCombSamples[:,-1] == np.max(gprnCombSamples[:,-1]))
gprnMapSample = gprnCombSamples[values,:].reshape(-1, 10)
nodes = [covfunc.QuasiPeriodic(gprnMapSample[-1,0], gprnMapSample[-1,1], 
                                    gprnMapSample[-1,2], gprnMapSample[-1,3])]
weight = [covfunc.SquaredExponential(gprnMapSample[-1,4], 
                                          gprnMapSample[-1,5])]
means = [meanfunc.Linear(gprnMapSample[-1,6], gprnMapSample[-1,7])]
jitter = [gprnMapSample[-1,8]]
GPRN = meanfield.inference(1, time, val1, val1err)
elbo, m, v = GPRN.ELBOcalc(nodes, weight, means, jitter, iterations = 50000,
                           mu='init', var='init')

tstar = np.linspace(time.min()-10, time.max()+10, 2500)
tstar = np.sort(np.concatenate((tstar, time)))
a, b, c = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v)
bmin, bmax = a-np.sqrt(b), a+np.sqrt(b)

axs[0].plot(tstar, a, '-b', alpha=0.75, label='GPRN')
axs[0].fill_between(tstar,  bmax[:,0], bmin[:,0], color="blue", alpha=0.25)
axs[0].legend(facecolor='whitesmoke', framealpha=1, 
              edgecolor='black', loc='upper right')

values = []
for i, j in enumerate(time):
    posVal = np.where(c == j)
    values.append(int(posVal[0]))
val1Pred = []
for i, j in enumerate(values):
    val1Pred.append(a[j])
gprnresiduals = val1 - np.array(np.squeeze(val1Pred))

from gprn import utils
rms = utils.wrms(gprnresiduals, val1err)

# textstr = 'rms = {0} m/s'.format(round(rms,3))
# props = dict(boxstyle='round', facecolor='whitesmoke', edgecolor='black')
# axs[1].text(0.425, 1.2, textstr, transform=axs[1].transAxes, fontsize=12,
#             verticalalignment='top', bbox=props)

axs[1].axhline(y=0, linestyle='--', color='k')
axs[1].set_ylabel('Residuals')#\nRMS: {0}m/s'.format(round(rms,3)))
axs[1].errorbar(time, gprnresiduals, val1err, fmt='.', color='blue')
axs[1].text(0.608, 0.935, 'RMS = {0}m/s'.format(round(rms, 3)),
            bbox={'facecolor':'whitesmoke', 'alpha':1, 'boxstyle':'round'},
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[1].transAxes, color='black', fontsize=10)
values = np.where(gpCombSamples[:,-1] == np.max(gpCombSamples[:,-1]))
gpMapSample = gpCombSamples[values,:].reshape(-1, 8)

kernel = ted.kernels.QuasiPeriodic(gpMapSample[-1,0], gpMapSample[-1,1], 
                               gpMapSample[-1,2], gpMapSample[-1,3])\
         + ted.kernels.WhiteNoise(gpMapSample[-1,6])
mean = ted.means.Linear(gpMapSample[-1,4], gpMapSample[-1,5])
tedibear = ted.process.GP(kernel, mean, time, val1, val1err)

tstar = np.linspace(time.min()-10, time.max()+10, 2500)
tstar = np.sort(np.concatenate((tstar, time)))
a, b, c = tedibear.prediction(kernel, mean, tstar)
bmin, bmax = a - b, a + b

axs[2].plot(tstar, a, '-r', alpha=1, label='GP')
axs[2].fill_between(tstar,  bmax.T, bmin.T, color="red", alpha=0.25)
axs[2].legend(facecolor='whitesmoke', framealpha=1, 
              edgecolor='black', loc='upper right')

values = []
for i, j in enumerate(time):
    posVal = np.where(c == j)
    values.append(int(posVal[0]))
val1Pred = []
for i, j in enumerate(values):
    val1Pred.append(a[j])
gpresiduals = val1 - np.array(val1Pred)

from tedi import utils
rms = utils.wrms(gpresiduals, val1err)

# textstr = 'rms = {0} m/s'.format(round(rms,3))
# props = dict(boxstyle='round', facecolor='whitesmoke', edgecolor='black')
# axs[3].text(0.425, 1.2, textstr, transform=axs[3].transAxes, fontsize=12,
#             verticalalignment='top', bbox=props)

axs[3].text(0.608, 0.935, 'RMS = {0}m/s'.format(round(rms, 3)),
            bbox={'facecolor':'whitesmoke', 'alpha':1, 'boxstyle':'round'},
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[3].transAxes, color='black', fontsize=10)
axs[3].axhline(y=0, linestyle='--', color='k')
axs[3].errorbar(time, gpresiduals, val1err, fmt='.', color='red')
axs[3].set_ylabel('Residuals')#\nRMS: {0}m/s'.format(round(rms,3)))
axs[3].set_xlabel('Time (BJD - 2400000)')

plt.tight_layout(pad=0.1, h_pad=0.25, w_pad=0.1)
plt.savefig('FWHMfit_withResidualsPlot.pdf', bbox_inches='tight')
plt.close('all')
