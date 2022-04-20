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

labels = np.array(["eta1", "eta2", "eta3", "eta4",
                   "eta11", "eta21","eta12", "eta22",
                   "slope1", "offset1","slope2", "offset2",
                   "jitter1","jitter2", "elbo"])
                   
time,rv,rverr,fw,fwerr= np.loadtxt("sunBinned_Dumusque.txt", skiprows = 1, 
                                    unpack = True,
                                    usecols = (0,1,2,9,10))

###### GP object #####
time, val1, val1err = time, rv, rverr
val2, val2err = fw, fwerr
GPRN = inference(1, time, val1, val1err, val2, val2err)

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
bmin1, bmax1 = a[0]-np.sqrt(b[0]), a[0]+np.sqrt(b[0])
bmin2, bmax2 = a[1]-np.sqrt(b[1]), a[1]+np.sqrt(b[1])

aa, bb, cc = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v,
                             separate=True)

values = []
for i, j in enumerate(time):
    posVal = np.where(c == j)
    values.append(int(posVal[0]))

val1Pred, val2Pred = [], []
for i, j in enumerate(values):
    val1Pred.append(a[0][j])
    val2Pred.append(a[1][j])

residuals1 = val1 - np.array(val1Pred)
residuals2 = val2 - np.array(val2Pred)

from gprn import utils
wrms0 = utils.wrms(val1 - val1.mean(), val1err)
wrms = utils.wrms(residuals1, val1err)
print(wrms0, wrms)
print('initWRMS / finalWRMS = {0}'.format(wrms0/wrms), file=f)
wrms0 = utils.wrms(val2 - val2.mean(), val2err)
wrms = utils.wrms(residuals2, val2err)
print(wrms0, wrms)
print('initWRMS / finalWRMS = {0}'.format(wrms0/wrms), file=f)
print(val1.ptp(), val2.ptp())
print(val1err.mean(), val2err.mean())
print('', file=f)
f.close()
