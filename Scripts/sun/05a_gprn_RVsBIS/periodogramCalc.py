import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.linewidth'] = 2
plt.close('all')
#from gprn import weightFunction, nodeFunction 
from gprn.covFunction import SquaredExponential, QuasiPeriodic
from gprn.meanFunction import Linear, Sine
from gprn.meanField import inference
import emcee

time,rv,rverr,bis,biserr= np.loadtxt("sunBinned_Dumusque.txt", skiprows = 1, 
                                    unpack = True,
                                    usecols = (0,1,2,7,8))
time, val1, val1err = time, rv, rverr
val2, val2err = bis, biserr
GPRN = inference(1, time, val1, val1err, val2, val2err)

filename =  "savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)
#autocorrelation
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.5 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
all_samples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)

#checking the likelihood that matters to us
values = np.where(all_samples[:,-1] == np.max(all_samples[:,-1]))
opt_samples = all_samples[values,:]
opt_samples = opt_samples.reshape(-1, 15)

nodes = [QuasiPeriodic(opt_samples[-1,0], opt_samples[-1,1], opt_samples[-1,2],
                       opt_samples[-1,3])]
weight = [SquaredExponential(opt_samples[-1,4], opt_samples[-1,5]),
          SquaredExponential(opt_samples[-1,6], opt_samples[-1,7])]
means = [Linear(opt_samples[-1,8], opt_samples[-1,9]),
         Linear(opt_samples[-1,10], opt_samples[-1,11])]
jitter = [opt_samples[-1,12], opt_samples[-1,13]]

elbo, m, v = GPRN.ELBOcalc(nodes, weight, means, jitter, 
                              iterations = 50000, mu='init', var='init')

tstar = np.linspace(time.min()-10, time.max()+10, 1000)
tstar = np.sort(np.concatenate((tstar, time)))
aa, bb, cc = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v, separate=True)

values = []
for i, j in enumerate(time):
    posVal = np.where(cc == j)
    values.append(int(posVal[0]))

val1Pred, val2Pred = [], []
nodo = []
for i, j in enumerate(values):
    val1Pred.append(aa[0][j])
    val2Pred.append(aa[1][j])
    nodo.append(bb[0,0][j])

residuals1 = val1 - np.array(val1Pred)
residuals2 = val2 - np.array(val2Pred)

plt.rcParams['figure.figsize'] = [8, 5]
fig, axs = plt.subplots(3,2)
axs[0,0].errorbar(time, residuals1, val1err, fmt='.', color='black')
axs[0,0].axhline(y=0, linestyle='--', color='b')
axs[0,0].set_ylabel('RV residuals (m/s)')
axs[0,0].tick_params(axis='both', which='both', labelbottom=False)
axs[1,0].errorbar(time, residuals2, val2err, fmt='.', color='black')
axs[1,0].axhline(y=0, linestyle='--', color='b')
axs[1,0].set_ylabel('BIS residuals (m/s)')
axs[1,0].tick_params(axis='both', which='both', labelbottom=False)
axs[2,0].plot(time, nodo, '.k')
axs[2,0].axhline(y=0, linestyle='--', color='b')
axs[2,0].set_ylabel('Node predictive')
axs[2,0].set_xlabel('Time (BJD - 2400000)')

from astropy.timeseries import LombScargle
linesize = 1

f1, p1 = LombScargle(time, residuals1, val1err).autopower()
axs[0,1].semilogx(1/f1, p1, color='black', linewidth=linesize)
axs[0,1].set_ylabel('Power')
bestf = f1[np.argmax(p1)]
bestp = 1/bestf
axs[0,1].axvline(x=13, ymin=p1.min(), ymax=100*p1.max(), color='red', alpha=0.75,
               linewidth=linesize)
axs[0,1].axvline(x=200, ymin=p1.min(), ymax=100*p1.max(), color='red', alpha=0.75,
               linewidth=linesize)
axs[0,1].tick_params(axis='both', which='both', labelbottom=False)

f2, p2 = LombScargle(time, residuals2, val2err).autopower()
axs[1,1].semilogx(1/f2, p2, color='black', linewidth=linesize)
axs[1,1].set_ylabel('Power')
bestf = f1[np.argmax(p2)]
bestp = 1/bestf
axs[1,1].axvline(x=13, ymin=p2.min(), ymax=100*p2.max(), color='red', alpha=0.75,
               linewidth=linesize)
axs[1,1].axvline(x=200, ymin=p2.min(), ymax=100*p2.max(), color='red', alpha=0.75,
               linewidth=linesize)
axs[1,1].tick_params(axis='both', which='both', labelbottom=False)

f3, p3 = LombScargle(time, nodo).autopower()
axs[2,1].semilogx(1/f3, p3, color='black', linewidth=linesize)
axs[2,1].set_ylabel('Power')
bestf = f1[np.argmax(p3)]
bestp = 1/bestf
axs[2,1].axvline(x=13, ymin=p3.min(), ymax=100*p3.max(), color='red', alpha=0.75,
               linewidth=linesize)
axs[2,1].axvline(x=200, ymin=p3.min(), ymax=100*p3.max(), color='red', alpha=0.75,
               linewidth=linesize)
axs[2,1].tick_params(axis='both', which='both', labelbottom=True)
axs[2,1].set_xlabel('Period (days)')

plt.savefig('residualsPeriodogram.pdf', bbox_inches='tight')
