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
from tedi import utils
import emcee 

from artgpn.arttwo import network
from artgpn import weight, node, mean

linesize = 1

################################################################################
rotP10700 = 34
data = np.loadtxt("10700_activity.csv",delimiter=',', skiprows=1, usecols=(1,4,5,11,12))
time = data[:,0].T
val1, val1err = data[:,1].T, data[:,2].T
val2, val2err = data[:,3].T, 2*val1err

filename =  "/home/camacho/GPRN/02_EXPRES/New/HD10700/02c_GP_RVsFWHM/savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)

#autocorrelation
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.1 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
combSamples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)
values = np.where(combSamples[:,-1] == np.max(combSamples[:,-1]))
mapSample = combSamples[values,:]
mapSample = mapSample.reshape(-1, 12)
nodes = [node.QuasiPeriodic(mapSample[-1,0], mapSample[-1,1], mapSample[-1,2])]
weights = [weight.Constant(mapSample[-1,3]**2),
           weight.Constant(mapSample[-1,4]**2)]
means = [mean.Linear(mapSample[-1,5], mapSample[-1,6]),
         mean.Linear(mapSample[-1,7], mapSample[-1,8])]
jitters = [mapSample[-1,9], mapSample[-1,10]]

GPnet = network(1, time, val1, val1err, val2, val2err)

tstar = np.linspace(time.min()-10, time.max()+10, 1000)
tstar = np.sort(np.concatenate((tstar, time)))

a1, b1, c1 = GPnet.prediction(nodes = nodes, weights = weights, means = means,
                              jitters = jitters,time = tstar, dataset = 1)
bmin1, bmax1 = a1 - b1, a1 + b1
a2, b2, c2 = GPnet.prediction(nodes = nodes, weights = weights, means = means,
                              jitters = jitters,time = tstar, dataset = 2)
bmin2, bmax2 = a2 - b2, a2 + b2

values = []
for i, j in enumerate(time):
    posVal = np.where(c1 == j)
    values.append(int(posVal[0]))
val1Pred = []
for i, j in enumerate(values):
    val1Pred.append(a1[j])
residuals1 = val1 - np.array(val1Pred)
values = []
for i, j in enumerate(time):
    posVal = np.where(c2 == j)
    values.append(int(posVal[0]))
val2Pred = []
for i, j in enumerate(values):
    val2Pred.append(a2[j])
residuals2 = val2 - np.array(val2Pred)

rms1 = utils.wrms(residuals1, val1err)
rms2 = utils.wrms(residuals2, val2err)

fig, axs = plt.subplots(nrows=4,ncols=1, sharex=True,
                        gridspec_kw={'width_ratios': [1],
                                     'height_ratios': [2.25, 1, 2.25, 1]})
fig.set_size_inches(w=7, h=4+4)
axs[0].fill_between(tstar,bmin1, bmax1, color="red", alpha=0.25)
axs[0].plot(tstar, a1, '-r', alpha=0.75, label='GP')
axs[0].errorbar(time,val1, val1err, fmt = "k.")
axs[0].set_ylabel("RV (m/s)")
axs[2].fill_between(tstar, bmin2, bmax2, color="red", alpha=0.25)
axs[2].plot(tstar, a2, '-r', alpha=0.75, label='GP')
axs[2].errorbar(time,val2, val2err, fmt = "k.")
axs[2].set_ylabel("FWHM (m/s)")

axs[1].axhline(y=0, linestyle='--', color='k')
axs[1].errorbar(time, residuals1, val1err, fmt = "k.")
axs[1].set_ylabel('Residuals (m/s)')
axs[3].axhline(y=0, linestyle='--', color='k')
axs[3].errorbar(time, residuals2, val2err, fmt = "k.")
axs[3].set_ylabel('Residuals (m/s)')

plt.tight_layout()
plt.savefig('GP_HD10700fitS.pdf', bbox_inches='tight')
# plt.close('all')
