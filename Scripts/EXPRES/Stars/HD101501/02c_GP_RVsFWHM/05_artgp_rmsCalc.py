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

from artgpn.arttwo import network
from artgpn import weight, node, mean
plt.rcParams['figure.figsize'] = [15, 5]

data = np.loadtxt("101501_activity.csv",delimiter=',', skiprows=1, usecols=(1,4,5,11,12))
time = data[:,0].T
val1, val1err = data[:,1].T, data[:,2].T
val2, val2err = data[:,3].T, 2*val1err

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

#checking the likelihood that matters to us
values = np.where(combSamples[:,-1] == np.max(combSamples[:,-1]))
mapSample = combSamples[values,:]
mapSample = mapSample.reshape(-1, 12)
#Having a solution from our MCMC we need to redefine all the network
nodes = [node.QuasiPeriodic(mapSample[-1,0], mapSample[-1,1], mapSample[-1,2])]
weights = [weight.Constant(mapSample[-1,3]**2),
           weight.Constant(mapSample[-1,4]**2)]
means = [mean.Linear(mapSample[-1,5], mapSample[-1,6]),
         mean.Linear(mapSample[-1,7], mapSample[-1,8])]
jitters = [mapSample[-1,9], mapSample[-1,10]]

GPnet = network(1, time, val1, val1err, val2, val2err)
tstar = np.linspace(time.min()-10, time.max()+10, 5000)
tstar = np.sort(np.concatenate((tstar, time)))

from artgpn import utils
a1, b1, c1 = GPnet.prediction(nodes = nodes, weights = weights, means = means,
                              jitters = jitters,time = tstar, dataset = 1)
bmin1, bmax1 = a1 - b1, a1 + b1

values = []
for i, j in enumerate(time):
    posVal = np.where(c1 == j)
    values.append(int(posVal[0]))
val1Pred = []
for i, j in enumerate(values):
    val1Pred.append(a1[j])
residuals1 = val1 - np.array(val1Pred)

wrms0 = utils.wrms(val1 - val1.mean(), val1err)
wrms = utils.wrms(residuals1, val1err)
print(wrms0,wrms)
print('initWRMS / finalWRMS = {0}'.format(wrms0/wrms), file=f)

a2, b2, c2 = GPnet.prediction(nodes = nodes, weights = weights, means = means,
                              jitters = jitters,time = tstar, dataset = 2)
bmin2, bmax2 = a2 - b2, a2 + b2

values = []
for i, j in enumerate(time):
    posVal = np.where(c2 == j)
    values.append(int(posVal[0]))
val2Pred = []
for i, j in enumerate(values):
    val2Pred.append(a2[j])
residuals2 = val2 - np.array(val2Pred)

wrms0 = utils.wrms(val2 - val2.mean(), val2err)
wrms = utils.wrms(residuals2, val2err)
print(wrms0,wrms)
print('initWRMS / finalWRMS = {0}'.format(wrms0/wrms), file=f)

print('', file=f)
f.close()
