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

labels = np.array(["$\eta_1$", "$\eta_2$", "$\eta_3$", "$\eta_4$",
                   "$\eta_11$", "$\eta_21$",
                   "slope", "offset",
                   "jitter", "logPost"])


time,fw,fwerr= np.loadtxt("sunBinned_Dumusque.txt", skiprows = 1, unpack = True,
                          usecols = (0,9,10))
time, val1, val1err = time, fw, fwerr
GPRN = inference(1, time, val1, val1err)

plt.rcParams['figure.figsize'] = [15, 5]
plt.errorbar(time,val1,val1err, fmt='.k', label='data')
plt.legend()
plt.title('Data')
plt.xlabel('Time (BJD-2400000)')
plt.ylabel('FWHM (m/s)')
plt.savefig('11_dataset.pdf', bbox_inches='tight')
plt.close('all')

filename =  "savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)
print(sampler.iteration)

storage_name = "10_results.txt"
f = open(storage_name, "a")
print('iterations:', sampler.iteration, file=f)

#autocorrelation
tau = sampler.get_autocorr_time(tol=100)
burnin = int(2*np.max(tau))
thin = int(0.1 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)

print("burn-in: {0}".format(burnin), file=f)
print("thin: {0}".format(thin), file=f)
print("flat chain shape: {0}".format(samples.shape), file=f)
print(file=f)
combSamples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)
values = np.where((combSamples[:,-1] > -1100.0))
combSamples = combSamples[values,:].reshape(-1, 10)

#plotting corner
import corner

labels = np.array(["$\eta_1$", "$\eta_2$", "$\eta_3$", "$\eta_4$",
                   "$\eta_11$", "$\eta_21$",
                   "slope", "offset",
                   "jitter", "logPost"])
corner.corner(combSamples, labels=labels, color="k", bins = 50,
              quantiles=[0.16, 0.5, 0.84], smooth=True, smooth1d=True, 
              show_titles=True, plot_density=True, plot_contours=True,
              fill_contours=True, plot_datapoints=True)
plt.savefig('12_cornerPlot_logP.pdf', bbox_inches='tight')
plt.close('all')

labels = np.array(["$\eta_1$", "$\eta_2$", "$\eta_3$", "$\eta_4$",
                   "$\eta_11$", "$\eta_21$",
                   "slope", "offset","jitter",])
corner.corner(combSamples [:,0:-1], labels=labels, color="k", bins = 50,
              quantiles=[0.16, 0.5, 0.84], smooth=True, smooth1d=True, 
              show_titles=True, plot_density=True, plot_contours=True,
              fill_contours=True, plot_datapoints=True)
plt.savefig('13_cornerPlot.pdf', bbox_inches='tight')
plt.close('all')

n1,n2,n3,n4,w11,w21,slp1,int1,j1,elbo = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
                            zip(*np.percentile(combSamples, [16, 50, 84],axis=0)))
print('*** Median values ***', file=f)
print('node eta1= {0[0]} +{0[1]} -{0[2]}'.format(n1), file=f)
print('node eta2= {0[0]} +{0[1]} -{0[2]}'.format(n2), file=f)
print('node eta3= {0[0]} +{0[1]} -{0[2]}'.format(n3), file=f)
print('node eta4= {0[0]} +{0[1]} -{0[2]}'.format(n4), file=f)
print('weight1 eta1= {0[0]} +{0[1]} -{0[2]}'.format(w11), file=f)
print('weight1 eta2= {0[0]} +{0[1]} -{0[2]}'.format(w21), file=f)
print('slope1= {0[0]} +{0[1]} -{0[2]}'.format(slp1), file=f)
print('offset1= {0[0]} +{0[1]} -{0[2]}'.format(int1), file=f)
print('jitter1= {0[0]} +{0[1]} -{0[2]}'.format(j1), file=f)
print('logPost= {0[0]} +{0[1]} -{0[2]}'.format(elbo), file=f)
print(file=f)

#checking the logPost that matters to us
values = np.where(combSamples[:,-1] == np.max(combSamples[:,-1]))
mapSample = combSamples[values,:]
mapSample = mapSample.reshape(-1, 10)
nodes = [QuasiPeriodic(mapSample[-1,0], mapSample[-1,1], mapSample[-1,2],
                       mapSample[-1,3])]
weight = [SquaredExponential(mapSample[-1,4], mapSample[-1,5])]
means = [Linear(mapSample[-1,6], mapSample[-1,7])]
jitter = [mapSample[-1,8]]

elbo, m, v = GPRN.ELBOcalc(nodes, weight, means, jitter, iterations = 50000,
                           mu='init', var='init')

tstar = np.linspace(time.min()-10, time.max()+10, 5000)
tstar = np.sort(np.concatenate((tstar, time)))
a, b, c = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v, 
                          variance=True)
bmin, bmax = a-np.sqrt(b), a+np.sqrt(b)

print('*** MAP values ***', file=f)
print('node eta1= {0}'.format(mapSample[-1,0]), file=f)
print('node eta2= {0}'.format(mapSample[-1,1]), file=f)
print('node eta3= {0}'.format(mapSample[-1,2]), file=f)
print('node eta4= {0}'.format(mapSample[-1,3]), file=f)
print('weight eta1= {0}'.format(mapSample[-1,4]), file=f)
print('weight eta2= {0}'.format(mapSample[-1,5]), file=f)
print('slope1 {0}'.format(mapSample[-1,6]), file=f)
print('offset= {0}'.format(mapSample[-1,7]), file=f)
print('jitter= {0}'.format(mapSample[-1,8]), file=f)
print('logPost= {0}'.format(mapSample[-1,9]), file=f)
print('done')

plt.rcParams['figure.figsize'] = [15, 1.5*5]
fig, axs = plt.subplots(2,1, sharex=True, gridspec_kw={'width_ratios': [1],
                                                       'height_ratios': [3, 1]})
axs[0].set_title('Fit')
axs[0].set_ylabel('FWHM (m/s)')
axs[0].errorbar(time, val1, val1err, fmt= '.k')
axs[0].plot(tstar, a[0,:].T, '-r')
axs[0].fill_between(tstar,  bmax[0].T, bmin[0].T, color="red", alpha=0.25)

values = []
for i, j in enumerate(time):
    posVal = np.where(c == j)
    values.append(int(posVal[0]))
val1Pred = []
for i, j in enumerate(values):
    val1Pred.append(a[0,j])
residuals = val1 - np.array(val1Pred)

from gprn import utils
rms = utils.wrms(residuals, val1err)

axs[1].set_title('Residuals (RMS: {0}m/s)'.format(np.round(rms,3)))
axs[1].set_ylabel('FWHM (m/s)')
axs[1].plot(time, residuals, '.k')
axs[1].axhline(y=0, linestyle='--', color='b')
axs[1].set_xlabel('Time (BJD - 2400000)')
plt.savefig('14_mapWithResidualsPlot.pdf', bbox_inches='tight')
plt.close('all')

print('RMS (m/s):', rms, file=f)
print(file=f)
f.close()


