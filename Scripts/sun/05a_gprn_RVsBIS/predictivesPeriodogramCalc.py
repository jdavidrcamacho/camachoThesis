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
from gprn.meanFunction import Linear, Sine
from gprn.meanField import inference
import emcee

labels = np.array(["eta1", "eta2", "eta3", "eta4",
                   "eta11", "eta21", "eta12", "eta22",
                   "slope1", "offset1", "sine1", "sine2", "sine3",
                   "slope2", "offset2", 
                   "jitter1", "jitter2", "elbo"])
                   
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

tstar = np.linspace(time.min()-10, time.max()+10, 5000)
tstar = np.sort(np.concatenate((tstar, time)))

a, b, c = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v, 
                          variance=True)
values = []
for i, j in enumerate(time):
    posVal = np.where(c == j)
    values.append(int(posVal[0]))

val1Pred, val2Pred = [], []
for i, j in enumerate(values):
    val1Pred.append(a[0][j])
    val2Pred.append(a[1][j])

rvFit = np.poly1d(np.polyfit(time, val1Pred, 1))
val1Pred = np.array(val1Pred)-rvFit(time)

bisFit = np.poly1d(np.polyfit(time, val2Pred, 1))
val2Pred = np.array(val2Pred)-bisFit(time)

fig, axs = plt.subplots(2,2)
axs[0,0].set_ylabel('RV (m/s)')
axs[0,0].errorbar(time, val1Pred, val1err, fmt= '.k')
axs[1,0].set_ylabel('BIS (m/s)')
axs[1,0].errorbar(time, val2Pred, val2err, fmt= '.k')
axs[1,0].set_xlabel('Time (BJD - 2400000)')
from astropy.timeseries import LombScargle
linesize = 1

f1, p1 = LombScargle(time, val1Pred, val1err).autopower()
axs[0,1].semilogx(1/f1, p1, color='black', linewidth=linesize)
axs[0,1].set_ylabel('Power')
bestf = f1[np.argmax(p1)]
bestp = 1/bestf
axs[0,1].axvline(x=13, ymin=p1.min(), ymax=100*p1.max(), color='red', alpha=0.75,
               linewidth=linesize)
axs[0,1].axvline(x=200, ymin=p1.min(), ymax=100*p1.max(), color='red', alpha=0.75,
               linewidth=linesize)
axs[0,1].tick_params(axis='both', which='both', labelbottom=False)

f2, p2 = LombScargle(time, val2Pred, val2err).autopower()
axs[1,1].semilogx(1/f2, p2, color='black', linewidth=linesize)
axs[1,1].set_ylabel('Power')
bestf = f1[np.argmax(p2)]
bestp = 1/bestf
axs[1,1].axvline(x=13, ymin=p2.min(), ymax=100*p2.max(), color='red', alpha=0.75,
               linewidth=linesize)
axs[1,1].axvline(x=200, ymin=p2.min(), ymax=100*p2.max(), color='red', alpha=0.75,
               linewidth=linesize)
axs[1,1].tick_params(axis='both', which='both', labelbottom=True)
axs[1,1].set_xlabel('Period (days)')
plt.savefig('predictivesPeriodogram.pdf', bbox_inches='tight')