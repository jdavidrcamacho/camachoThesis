import numpy as np 
import emcee
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')
from tedi import process, kernels, means, utils

data = np.loadtxt("26965_activity.csv",delimiter=',', 
                    skiprows=1, usecols=(1,4,5,11,12))
time = data[:,0].T
val1, val1err = data[:,3].T, 2*data[:,2]

filename =  "savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)
storage_name = "10_results.txt"
f = open(storage_name, "a")

#autocorrelation
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.1 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
combSamples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)

#checking the logprob that matters to us
values = np.where(combSamples[:,-1] == np.max(combSamples[:,-1]))
mapSample = combSamples[values,:]
mapSample = mapSample.reshape(-1, 8)

kernel = kernels.QuasiPeriodic(mapSample[-1,0], mapSample[-1,1], 
                               mapSample[-1,2], mapSample[-1,3])\
        +kernels.WhiteNoise(mapSample[-1,6])
mean = means.Linear(mapSample[-1,4], mapSample[-1,5])

tedibear = process.GP(kernel, mean, time, val1, val1err)

tstar = np.linspace(time.min()-10, time.max()+10, 5000)
tstar = np.sort(np.concatenate((tstar, time)))
a, b, c = tedibear.prediction(kernel, mean, tstar)
values = []
for i, j in enumerate(time):
    posVal = np.where(c == j)
    values.append(int(posVal[0]))
val1Pred = []
for i, j in enumerate(values):
    val1Pred.append(a[j])
residuals = val1 - np.array(val1Pred)

rms0 = utils.rms(val1 - mean(time))
rms = utils.rms(residuals)
print('initRMS / finalRMS = {0}'.format(rms0/rms), file=f)

wrms0 = utils.wrms(val1 - mean(time), val1err)
wrms = utils.wrms(residuals, val1err)
print('initWRMS / finalWRMS = {0}'.format(wrms0/wrms), file=f)

f.close()
