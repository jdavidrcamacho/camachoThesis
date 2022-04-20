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

labels = np.array(["$\eta_1$", "$\eta_2$", "$\eta_3$", "$\eta_4$",
                   "$\eta_11$", "$\eta_21$",
                   "slope", "offset",
                   "jitter", "logPost"])


time,fw,fwerr= np.loadtxt("sunBinned_Dumusque.txt", skiprows = 1, unpack = True,
                          usecols = (0,9,10))
time, val1, val1err = time, fw, fwerr

filename =  "savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)
print(sampler.iteration)

storage_name = "10_results.txt"
f = open(storage_name, "a")
print('iterations:', sampler.iteration, file=f)

#autocorrelation
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.1 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)

print("burn-in: {0}".format(burnin), file=f)
print("thin: {0}".format(thin), file=f)
print("flat chain shape: {0}".format(samples.shape), file=f)
print(file=f)
combSamples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)
values = np.where((combSamples[:,-1] > -1100.0))
combSamples = combSamples[values,:].reshape(-1, 10)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.linewidth'] = 2
combSamples[:,0] = np.log(combSamples[:,0])
combSamples[:,4] = np.log(combSamples[:,4])
import corner
import matplotlib 
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 
labels = np.array(["$\log(\eta_1^{n})$", "$\eta_2^{n}$", "$\eta_3^{n}$", "$\eta_4^{n}$",
                   "$\log(\eta_1^{w_{FWHM}})$", "$\eta_2^{w_{FWHM}}$",
                   "slope", "offset","s",])
corner.corner(combSamples [:,0:-1], labels=labels, color="k", bins = 50,
              top_ticks=True,
              label_kwargs={"fontsize": 34}, max_n_ticks=3,
              smooth=True, smooth1d=True, 
              plot_density=True, plot_contours=True,
              fill_contours=True, plot_datapoints=True)
plt.savefig('FWHM_gprn_cornerPlot.pdf', bbox_inches='tight')
plt.close('all')

