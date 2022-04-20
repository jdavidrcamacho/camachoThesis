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
                   
data = np.loadtxt("34411_activity.csv",delimiter=',', skiprows=1, usecols=(1,4,5,11,12))
time = data[:,0].T
#rv
val1, val1err = data[:,1].T, data[:,2].T
#fwhm
val3, val3err = data[:,3].T, 2*val1err
GPRN = inference(1, time, val1, val1err, val3,val3err)

filename =  "savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)

#autocorrelation
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.1 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)

#samples
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
all_samples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)

#checking the logprob that matters to us
values = np.where(all_samples[:,-1] == np.max(all_samples[:,-1]))
opt_samples = all_samples[values,:]
opt_samples = opt_samples.reshape(-1, 15)

#nodes, weights, and jitters
nodes = [QuasiPeriodic(opt_samples[-1,0], opt_samples[-1,1], opt_samples[-1,2],
                       opt_samples[-1,3])]
weight = [SquaredExponential(opt_samples[-1,4], opt_samples[-1,5]),
          SquaredExponential(opt_samples[-1,6], opt_samples[-1,7])]
means = [Linear(opt_samples[-1,8], opt_samples[-1,9]),
         Linear(opt_samples[-1,10], opt_samples[-1,11])]
jitter = [opt_samples[-1,12], opt_samples[-1,13]]

elbo, m, v = GPRN.ELBOcalc(nodes, weight, means, jitter, 
                           iterations = 50000, mu='init', var='init')
RV_A, _,_ = GPRN.Prediction(nodes,weight, means, jitter, time, m, v)
RV_A = RV_A[0]
RV_C = val1 - RV_A  

eRV_C = val1err
eRV_A = ['NaN' for i in val1err]

print(time.shape, RV_C.shape, eRV_C.shape, RV_A.shape, np.array(eRV_A).shape)
results = np.stack((time, RV_C, eRV_C, RV_A, np.array(eRV_A)))
results = results.T

star=34411
group_name='Porto'
method_name='GPRN'

header= 'Time[MJD],RV_C[m/s],eRV_C[m/s],RV_A[m/s],eRV_A[m/s]'
fmt = '%s'
np.savetxt('{0}_{1}_{2}_results.csv'.format(star,group_name,method_name), 
           results, delimiter=',',header=header, fmt=fmt, comments='')
