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



#################### GP ########################################################
from tedi import process, kernels, means

time,rv,rverr = np.loadtxt("/home/camacho/GPRN/01_SUN/65_sun/GP/01b_GP_RV/sunBinned_Dumusque.txt", 
                           skiprows = 1, unpack = True, usecols = (0,1,2))
val1, val1err = rv, rverr
kernel = kernels.QuasiPeriodic(1, 2, 3, 4) + kernels.WhiteNoise(0.1)
mean = means.Linear(0, -20)
tedibear = process.GP(kernel, mean, time, val1, val1err)

tstar = np.linspace(time.min()-1, time.max()+1, 1000)

filename =  "/home/camacho/GPRN/01_SUN/65_sun/GP/01b_GP_RV/savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)
tau = sampler.get_autocorr_time(tol=25)
burnin = int(2*np.max(tau))
thin = int(0.1 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
combSamples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)
values = np.where((combSamples[:,-1] > -800.0))
gpSamples = combSamples[values,:].reshape(-1, 8)

# MAP values
values = np.where(gpSamples[:,-1] == np.max(gpSamples[:,-1]))
mapSample = gpSamples[values,:].reshape(-1, 8)

kernel = kernels.QuasiPeriodic(mapSample[-1,0], mapSample[-1,1], 
                               mapSample[-1,2], mapSample[-1,3])\
        +kernels.WhiteNoise(mapSample[-1,6])
mean = means.Linear(mapSample[-1,4], mapSample[-1,5])
tedibear = process.GP(kernel, mean, time, val1, val1err)
m,s,_ = tedibear.prediction(kernel, mean, tstar)

fig, axs = plt.subplots(nrows = 1, ncols = 2, sharex=True, sharey=True)
fig.set_size_inches(w=10, h=5)
axs[0].errorbar(time, val1, val1err, fmt= '.k')
axs[0].plot(tstar, m, '-r')
axs[0].fill_between(tstar,  m+s, m-s, color="red", alpha=0.25)
axs[0].set_xlabel('Time (BJD-2400000)')
axs[0].set_ylabel('RV (m/s)')


################# GPRN #########################################################
from gprn.covFunction import SquaredExponential, QuasiPeriodic
from gprn.meanFunction import Linear
from gprn.meanField import inference

time,rv,rverr= np.loadtxt("/home/camacho/GPRN/01_SUN/65_sun/GPRN/01a_gprn_RV/sunBinned_Dumusque.txt", 
                          skiprows = 1, unpack = True, usecols = (0,1,2))
time, val1, val1err = time, rv, rverr
GPRN = inference(1, time, val1, val1err)

filename = "/home/camacho/GPRN/01_SUN/65_sun/GPRN/01a_gprn_RV/savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.1 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
combSamples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)
values = np.where((combSamples[:,-1] > -850.0))
gprnSamples = combSamples[values,:].reshape(-1, 10)

# MAP values
values = np.where(gprnSamples[:,-1] == np.max(gprnSamples[:,-1]))
mapSample = gprnSamples[values,:].reshape(-1, 10)

nodes = [QuasiPeriodic(mapSample[-1,0], mapSample[-1,1], mapSample[-1,2], mapSample[-1,3])]
weight = [SquaredExponential(mapSample[-1,4], mapSample[-1,5])]
means = [Linear(mapSample[-1,6], mapSample[-1,7])]
jitter = [mapSample[-1,8]]

elbo, m, v = GPRN.ELBOcalc(nodes, weight, means, jitter, iterations = 50000, mu='init', var='init')
a, b = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v, variance= True)
bmin, bmax = a-np.sqrt(b), a+np.sqrt(b)

axs[1].errorbar(time, val1, val1err, fmt= '.k')
axs[1].set_xlabel('Time (BJD-2400000)')
axs[1].plot(tstar, a[0].T, '-r')
axs[1].fill_between(tstar,  bmax[0].T, bmin[0].T, color="red", alpha=0.25)
