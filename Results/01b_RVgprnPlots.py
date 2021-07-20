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

time,rv,rverr = np.loadtxt("/home/camacho/Github/camachoThesis/sunData/sunBinned_Dumusque.txt", 
                           skiprows = 1, unpack = True, usecols = (0,1,2))
val1, val1err = rv, rverr
GPRN = inference(1, time, val1, val1err)

filename =  "/home/camacho/GPRN/01_SUN/65_sun/GPRN/01a_gprn_RV/savedProgress.h5"
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

#checking the logPost that matters to us
values = np.where(combSamples[:,-1] == np.max(combSamples[:,-1]))
mapSample = combSamples[values,:]
mapSample = mapSample.reshape(-1, 10)
nodes = [QuasiPeriodic(mapSample[-1,0], mapSample[-1,1], mapSample[-1,2],
                       mapSample[-1,3])]
weight = [SquaredExponential(mapSample[-1,4], mapSample[-1,5])]
means = [Linear(mapSample[-1,6], mapSample[-1,7])]
jitter = [mapSample[-1,8]]

elbo, m, v = GPRN.ELBOcalc(nodes, weight, means, jitter, iterations = 50000,
                           mu='init', var='init')
vals, _ = GPRN.Prediction(nodes, weight, means, jitter, time, 
                       m, v, variance= True)
residuals = val1 - vals[0]
rms = np.sqrt(np.sum((residuals - np.mean(residuals))**2)/time.size)

tstar = np.linspace(time.min()-10, time.max()+10, 10000)

a, b = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v, variance= True)
bmin, bmax = a-np.sqrt(b), a+np.sqrt(b)

aa, bb = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v, separate=True)

fig = plt.figure(constrained_layout=True, figsize=(7, 5))
axs = fig.subplot_mosaic(
    [
        ['predictive', 'node'],
        ['predictive', 'weight'],
    ],
)
axs['predictive'].set(xlabel='Time (BJD - 2450000)', ylabel='RV (m/s)')
axs['predictive'].errorbar(time, val1, val1err, fmt= '.k')
axs['predictive'].plot(tstar, a[0].T, '-r')
axs['predictive'].fill_between(tstar,  bmax[0].T, bmin[0].T, color="red", alpha=0.25)
axs['node'].set(xlabel='', ylabel='node')
axs['node'].plot(tstar, bb[0].T, '-b')
axs['weight'].set(xlabel='Time (BJD - 2450000)', ylabel='weight (m/s)')
axs['weight'].plot(tstar, bb[1].T, '-b')

plt.tight_layout()
fig.savefig('RVfit_GPRN.pdf', bbox_inches='tight')
