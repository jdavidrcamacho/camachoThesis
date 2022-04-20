import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.linewidth'] = 2

import emcee
import corner
labels = np.array(["$\eta_1$", "$\eta_2$", "$\eta_3$", "$\eta_4$",
                   "$\eta_11$", "$\eta_21$",
                   "slope", "offset",
                   "jitter", "logPost"])

time,rv,rverr= np.loadtxt("sunBinned_Dumusque.txt", skiprows = 1, 
                          unpack = True, usecols = (0,1,2))
time, val1, val1err = time, rv, rverr

filename =  "savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)

#autocorrelation
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.1 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
combSamples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)
values = np.where((combSamples[:,-1] > -850.0))
combSamples = combSamples[values,:].reshape(-1, 10)


combSamples[:,0] = np.log(combSamples[:,0])
combSamples[:,4] = np.log(combSamples[:,4])
import corner
import matplotlib 
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 
labels = np.array(["$\log(\eta_1^{n})$", "$\eta_2^{n}$", "$\eta_3^{n}$", "$\eta_4^{n}$",
                   "$\log(\eta_1^{w_{RV}})$", "$\eta_2^{w_{RV}}$",
                   "slope", "offset","s",])
corner.corner(combSamples [:,0:-1], labels=labels, color="k", bins = 50,
              top_ticks=True,
              label_kwargs={"fontsize": 34}, max_n_ticks=3,
              smooth=True, smooth1d=True, 
              plot_density=True, plot_contours=True,
              fill_contours=True, plot_datapoints=True)
plt.savefig('RV_gprn_cornerPlot.pdf', bbox_inches='tight')
plt.close('all')


