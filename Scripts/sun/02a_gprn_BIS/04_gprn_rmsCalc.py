import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')

from gprn.covFunction import SquaredExponential, QuasiPeriodic
from gprn.meanFunction import Linear
from gprn.meanField import inference
from gprn import utils

import emcee

time,bis,biserr= np.loadtxt("sunBinned_Dumusque.txt", skiprows = 1, unpack = True,
                          usecols = (0,7,8))
time, val1, val1err = time, bis, biserr
GPRN = inference(1, time, val1, val1err)


filename =  "savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)

#autocorrelation
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.1 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
logProbSamples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
combSamples = np.concatenate((samples, logProbSamples[:, None]), axis=1)

#checking the logPost that matters to us
values = np.where(combSamples[:,-1] == np.max(combSamples[:,-1]))
mapSample = combSamples[values,:]
mapSample = mapSample.reshape(-1, 10)
nodes = [QuasiPeriodic(mapSample[-1,0], mapSample[-1,1], mapSample[-1,2],
                       mapSample[-1,3])]
weight = [SquaredExponential(mapSample[-1,4], mapSample[-1,5])]
means = [Linear(mapSample[-1,6], mapSample[-1,7])]
jitter = [mapSample[-1,8]]

elbo, m, v = GPRN.ELBOcalc(nodes, weight, means, jitter, 
                              iterations = 50000, mu='init', var='init')
tstar = np.linspace(time.min()-10, time.max()+10, 5000)
tstar = np.sort(np.concatenate((tstar, time)))

a, b, c = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v, 
                          variance=True)
bmin1, bmax1 = a[0]-np.sqrt(b[0]), a[0]+np.sqrt(b[0])

aa, bb, cc = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v,
                             separate=True)

values = []
for i, j in enumerate(time):
    posVal = np.where(c == j)
    values.append(int(posVal[0]))

val1Pred  = []
for i, j in enumerate(values):
    val1Pred.append(a[0][j])

residuals1 = val1 - np.array(val1Pred)

wrms0 = utils.wrms(val1 - val1.mean(), val1err)
wrms = utils.wrms(val1 - np.array(val1Pred), val1err)

storage_name = "10_results.txt"
f = open(storage_name, "a")
print('initWRMS / finalWRMS = {0}'.format(wrms0/wrms), file=f)
f.close()
