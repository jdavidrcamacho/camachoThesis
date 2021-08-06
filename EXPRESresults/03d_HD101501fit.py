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

linesize = 1

################################################################################
data = np.loadtxt("101501_activity.csv",delimiter=',', skiprows=1, usecols=(1,4,5,11,12))
time1 = data[:,0].T
val1RV, val1RVerr = data[:,1].T, data[:,2].T
val1FW, val1FWerr = data[:,3].T, 2*val1RVerr
GPRN1 = inference(1, time1, val1RV, val1RVerr, val1FW, val1FWerr)

tstar = np.linspace(time1.min()-10, time1.max()+10, 10000)

filename = "/home/camacho/GPRN/02_EXPRES/New/HD101501/HD101501_RVsFW/savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)
#autocorrelation
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.1 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
samples1 = np.concatenate((samples, log_prob_samples[:, None]), axis=1)
#checking the likelihood that matters to us
values = np.where(samples1[:,-1] == np.max(samples1[:,-1]))
MAPsamples1 = samples1[values,:].reshape(-1, 15)


nodes = [QuasiPeriodic(MAPsamples1[-1,0], MAPsamples1[-1,1], MAPsamples1[-1,2],
                       MAPsamples1[-1,3])]
weight = [SquaredExponential(MAPsamples1[-1,4], MAPsamples1[-1,5]),
          SquaredExponential(MAPsamples1[-1,6], MAPsamples1[-1,7])]
means = [Linear(MAPsamples1[-1,8], MAPsamples1[-1,9]),
         Linear(MAPsamples1[-1,10], MAPsamples1[-1,11])]
jitter = [MAPsamples1[-1,12], MAPsamples1[-1,13]]
GPRN = inference(1, time1, val1RV, val1RVerr, val1FW, val1FWerr)
elbo, vm, vv = GPRN.ELBOcalc(nodes, weight, means, jitter, iterations = 5000,
                             mu='init', var='init')

elbo, m, v = GPRN.ELBOcalc(nodes, weight, means, jitter, 
                              iterations = 50000, mu='init', var='init')

tstar = np.linspace(time1.min()-10, time1.max()+10, 5000)
tstar = np.sort(np.concatenate((tstar, time1)))

a, b, c = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v, 
                          variance=True)
bmin1, bmax1 = a[0]-np.sqrt(b[0]), a[0]+np.sqrt(b[0])
bmin2, bmax2 = a[1]-np.sqrt(b[1]), a[1]+np.sqrt(b[1])

aa, bb, cc = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v,
                             separate=True)

fig = plt.figure(constrained_layout=True, figsize=(7, 4))
axs = fig.subplot_mosaic(
    [
        ['predictive', 'node'],
        ['predictive', 'weight'],
    ],
)
axs['predictive'].set(xlabel='Time (MJD)', ylabel='RV (m/s)')
axs['predictive'].errorbar(time1, val1RV, val1RVerr, fmt= '.k')
axs['predictive'].plot(tstar, a[0].T, '-r')
axs['predictive'].fill_between(tstar,  bmax1.T, bmin1.T, color="red", alpha=0.25)
axs['node'].set(xlabel='', ylabel='Node')
axs['node'].plot(tstar, bb[0,0].T, '-b')
axs['node'].tick_params(axis='both', which='both', labelbottom=False)
axs['weight'].set(xlabel='Time (MJD)', ylabel='Weight (m/s)')
axs['weight'].plot(tstar, bb[1,0].T, '-b')
plt.tight_layout()
plt.savefig('HD101501_fit.pdf', bbox_inches='tight')
plt.close('all')