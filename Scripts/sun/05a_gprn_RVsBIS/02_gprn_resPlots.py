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
                   
time,rv,rverr,bis,biserr= np.loadtxt("sunBinned_Dumusque.txt", skiprows = 1, 
                                    unpack = True,
                                    usecols = (0,1,2,7,8))

###### GP object #####
time, val1, val1err = time, rv, rverr
val2, val2err = bis, biserr
GPRN = inference(1, time, val1, val1err, val2, val2err)

plt.rcParams['figure.figsize'] = [15, 5]
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
axs[0].errorbar(time, val1, val1err, fmt='.k', label='RV')
axs[1].errorbar(time, val2, val2err, fmt='.k', label='FWHM')
axs[1].set_xlabel('Time (BJD-2400000)')
axs[0].set_ylabel('RV (m/s)')
axs[1].set_ylabel('BIS (m/s)')
plt.title('Data')
plt.savefig('11_datasets.pdf', bbox_inches='tight')
plt.close('all')
print('11_dataset.pdf done...')

filename =  "savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)
print(sampler.iteration)

storage_name = "10_results.txt"
f = open(storage_name, "a")
print('iterations:', sampler.iteration, file=f)

#autocorrelation
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.1 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)

print("burn-in: {0}".format(burnin), file=f)
print("thin: {0}".format(thin), file=f)
print("flat chain shape: {0}".format(samples.shape), file=f)
all_samples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)

#plotting corner
import corner

labels = np.array(["eta1", "eta2", "eta3", "eta4",
                   "eta11", "eta21","eta12", "eta22",
                   "slope1", "offset1","slope2", "offset2",
                   "jitter1","jitter2",
                   "logPost"])
corner.corner(all_samples, labels=labels, color="k", bins = 50,
              quantiles=[0.16, 0.5, 0.84], smooth=True, smooth1d=True, 
              show_titles=True, plot_density=True, plot_contours=True,
              fill_contours=True, plot_datapoints=True)
plt.savefig('12_cornerPlot_logP.pdf', bbox_inches='tight')
plt.close('all')

labels = np.array(["eta1", "eta2", "eta3", "eta4",
                   "eta11", "eta21","eta12", "eta22",
                   "slope1", "offset1","slope2", "offset2",
                   "jitter1","jitter2"])
corner.corner(all_samples[:,0:-1], labels=labels, color="k", bins = 50,
              quantiles=[0.16, 0.5, 0.84], smooth=True, smooth1d=True, 
              show_titles=True, plot_density=True, plot_contours=True,
              fill_contours=True, plot_datapoints=True)
plt.savefig('13_cornerPlot.pdf', bbox_inches='tight')
plt.close('all')

n1,n2,n3,n4,w11,w21,w12,w22,slp1,int1,slp2,int2,j1,j2,elbo = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
                            zip(*np.percentile(all_samples, [16, 50, 84],axis=0)))
print('Medians', file=f)
print(file=f)
print('node eta1= {0[0]} +{0[1]} -{0[2]}'.format(n1), file=f)
print('node eta2= {0[0]} +{0[1]} -{0[2]}'.format(n2), file=f)
print('node eta3= {0[0]} +{0[1]} -{0[2]}'.format(n3), file=f)
print('node eta4= {0[0]} +{0[1]} -{0[2]}'.format(n4), file=f)
print(file=f)
print('weight1 eta1= {0[0]} +{0[1]} -{0[2]}'.format(w11), file=f)
print('weight1 eta2= {0[0]} +{0[1]} -{0[2]}'.format(w21), file=f)
print('weight2 eta1= {0[0]} +{0[1]} -{0[2]}'.format(w12), file=f)
print('weight2 eta2= {0[0]} +{0[1]} -{0[2]}'.format(w22), file=f)
print(file=f)
print('slope1= {0[0]} +{0[1]} -{0[2]}'.format(slp1), file=f)
print('offset1= {0[0]} +{0[1]} -{0[2]}'.format(int1), file=f)
print('slope2= {0[0]} +{0[1]} -{0[2]}'.format(slp2), file=f)
print('offset2= {0[0]} +{0[1]} -{0[2]}'.format(int2), file=f)
print(file=f)
print('jitter1= {0[0]} +{0[1]} -{0[2]}'.format(j1), file=f)
print('jitter2= {0[0]} +{0[1]} -{0[2]}'.format(j2), file=f)
print(file=f)
print('logPost= {0[0]} +{0[1]} -{0[2]}'.format(elbo), file=f)
print(file=f)
nodes = [QuasiPeriodic(n1[0], n2[0], n3[0],n4[0])]
weight = [SquaredExponential(w11[0], w21[0]),
          SquaredExponential(w12[0], w22[0])]
means = [Linear(slp1[1], int1[0]), Linear(slp2[1], int2[0])]
jitter = [j1[0], j2[0]]

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

print('MAP values', file=f)
print(file=f)
print('node eta1= {0}'.format(opt_samples[-1,0]), file=f)
print('node eta2= {0}'.format(opt_samples[-1,1]), file=f)
print('node eta3= {0}'.format(opt_samples[-1,2]), file=f)
print('node eta4= {0}'.format(opt_samples[-1,3]), file=f)
print(file=f)
print('weight1 eta1= {0}'.format(opt_samples[-1,4]), file=f)
print('weight1 eta2= {0}'.format(opt_samples[-1,5]), file=f)
print('weight1 eta1= {0}'.format(opt_samples[-1,6]), file=f)
print('weight1 eta2= {0}'.format(opt_samples[-1,7]), file=f)
print(file=f)
print('slope1= {0}'.format(opt_samples[-1,8]), file=f)
print('offset1= {0}'.format(opt_samples[-1,9]), file=f)
print('slope2= {0}'.format(opt_samples[-1,10]), file=f)
print('offset2= {0}'.format(opt_samples[-1,11]), file=f)
print(file=f)
print('jitter1= {0}'.format(opt_samples[-1,12]), file=f)
print('jitter2= {0}'.format(opt_samples[-1,13]), file=f)
print(file=f)
print('logPost= {0}'.format(opt_samples[-1,14]), file=f)
print('done')

from gprn import utils

plt.rcParams['figure.figsize'] = [15, 1.5*5]
fig, axs = plt.subplots(2,1, sharex=True, gridspec_kw={'width_ratios': [1],
                                                       'height_ratios': [3, 1]})
axs[0].set_ylabel('RV (m/s)')
axs[0].errorbar(time, val1, val1err, fmt= '.k')
axs[0].plot(tstar, a[0].T, '-r')
axs[0].fill_between(tstar,  bmax1.T, bmin1.T, color="red", alpha=0.25)
rms = utils.wrms(residuals1, val1err)
axs[1].set_title('Residuals (RMS: {0}m/s)'.format(np.round(rms,3)))
axs[1].set_ylabel('RV (m/s)')
axs[1].plot(time, residuals1, '.k')
axs[1].axhline(y=0, linestyle='--', color='b')
axs[1].set_xlabel('Time (BJD - 2400000)')
plt.savefig('14_mapWithResidualsPlot_RV.pdf', bbox_inches='tight')
plt.close('all')
print('RVs RMS (m/s):', rms, file=f)
print(file=f)

plt.rcParams['figure.figsize'] = [15, 1.5*5]
fig, axs = plt.subplots(2,1, sharex=True, gridspec_kw={'width_ratios': [1],
                                                       'height_ratios': [3, 1]})
axs[0].set_ylabel('BIS (m/s)')
axs[0].errorbar(time, val2, val2err, fmt= '.k')
axs[0].plot(tstar, a[1].T, '-r')
axs[0].fill_between(tstar,  bmax2.T, bmin2.T, color="red", alpha=0.25)
rms = utils.wrms(residuals2, val2err)
axs[1].set_title('Residuals (RMS: {0}m/s)'.format(np.round(rms,3)))
axs[1].set_ylabel('BIS (m/s)')
axs[1].plot(time, residuals2, '.k')
axs[1].axhline(y=0, linestyle='--', color='b')
axs[1].set_xlabel('Time (BJD - 2400000)')
plt.savefig('15_mapWithResidualsPlot_BIS.pdf', bbox_inches='tight')
plt.close('all')
print('BIS RMS (m/s):', rms, file=f)
print(file=f)



