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
import corner
from tedi import process, kernels, means

data = np.loadtxt("101501_activity.csv",delimiter=',', 
                    skiprows=1, usecols=(1,4,5,11,12))
time = data[:,0].T
val1, val1err = data[:,3].T, 2*data[:,2]


plt.rcParams['figure.figsize'] = [15, 5]
plt.figure()
plt.errorbar(time, val1, val1err, fmt='.k')
plt.title('Data')
plt.xlabel('Time (BJD-2400000)')
plt.ylabel('FWHM (m/s)')
plt.savefig('11_dataset.pdf', bbox_inches='tight')
plt.close('all')

filename =  "savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)

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
combSamples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)

labels = np.array(["$\eta_1$", "$\eta_2$", "$\eta_3$", "$\eta_4$", "slope", 
                    "offset", "s", "logPost"])
corner.corner(combSamples, labels=labels, color="k", bins = 50,
              quantiles=[0.16, 0.5, 0.84], smooth=True, smooth1d=True, 
              show_titles=True, plot_density=True, plot_contours=True,
              fill_contours=True, plot_datapoints=True)
plt.savefig('12_cornerPlot_logPost.pdf', bbox_inches='tight')
plt.close('all')

labels = np.array(["$\eta_1$", "$\eta_2$", "$\eta_3$", "$\eta_4$", "slope", 
                    "offset", "s"])
corner.corner(combSamples[:,:-1], labels=labels, color="k", bins = 50,
              quantiles=[0.16, 0.5, 0.84], smooth=True, smooth1d=True, 
              show_titles=True, plot_density=True, plot_contours=True,
              fill_contours=True, plot_datapoints=True)
plt.savefig('13_cornerPlot.pdf', bbox_inches='tight')
plt.close('all')

e1,e2,e3,e4,sl,intr,s,logl = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
                                   zip(*np.percentile(combSamples, [16, 50, 84],axis=0)))
print('*** Median values ***', file=f)
print('eta1= {0[0]} +{0[1]} -{0[2]}'.format(e1), file=f)
print('eta2= {0[0]} +{0[1]} -{0[2]}'.format(e2), file=f)
print('eta3= {0[0]} +{0[1]} -{0[2]}'.format(e3), file=f)
print('eta4= {0[0]} +{0[1]} -{0[2]}'.format(e4), file=f)
print('slope= {0[0]} +{0[1]} -{0[2]}'.format(sl), file=f)
print('offset= {0[0]} +{0[1]} -{0[2]}'.format(intr), file=f)
print('s= {0[0]} +{0[1]} -{0[2]}'.format(s), file=f)
print('logPost= {0[0]} +{0[1]} -{0[2]}'.format(logl), file=f)
print(file=f)

#checking the logprob that matters to us
values = np.where(combSamples[:,-1] == np.max(combSamples[:,-1]))
mapSample = combSamples[values,:]
mapSample = mapSample.reshape(-1, 8)

kernel = kernels.QuasiPeriodic(mapSample[-1,0], mapSample[-1,1], 
                               mapSample[-1,2], mapSample[-1,3])\
        +kernels.WhiteNoise(mapSample[-1,6])
mean = means.Linear(mapSample[-1,4], mapSample[-1,5])

tedibear = process.GP(kernel, mean, time, val1, val1err)
tstar = np.linspace(time.min()-10, time.max()+10, 5000)
tstar = np.sort(np.concatenate((tstar, time)))
a, b, c = tedibear.prediction(kernel, mean, tstar)
bmin, bmax = a - b, a + b

plt.figure()
plt.errorbar(time, val1, val1err, fmt= '.k')
plt.plot(tstar, a, '-r')
plt.fill_between(tstar, bmin, bmax, color="red", alpha=0.25)
plt.xlabel('Time (BJD-2400000)')
plt.ylabel('FWHM (m/s)')
plt.title('MAP values fit')
plt.savefig('14_mapPlot.pdf', bbox_inches='tight')
plt.close('all')

print('MAP values', file=f)
print('eta1= {0}'.format(mapSample[-1,0]), file=f)
print('eta2= {0}'.format(mapSample[-1,1]), file=f)
print('eta3= {0}'.format(mapSample[-1,2]), file=f)
print('eta4= {0}'.format(mapSample[-1,3]), file=f)
print('slope= {0}'.format(mapSample[-1,4]), file=f)
print('offset= {0}'.format(mapSample[-1,5]), file=f)
print('s= {0}'.format(mapSample[-1,6]), file=f)
print('logPost= {0}'.format(mapSample[-1,7]), file=f)
print(file=f)

plt.rcParams['figure.figsize'] = [15, 1.5*5]
fig, axs = plt.subplots(2,1, sharex=True, gridspec_kw={'width_ratios': [1],
                                                       'height_ratios': [3, 1]})
axs[0].set_title('Fit')
axs[0].set_ylabel('FWHM (m/s)')
axs[0].errorbar(time, val1, val1err, fmt= '.k')
axs[0].plot(tstar, a, '-r')
axs[0].fill_between(tstar, bmin, bmax, color="red", alpha=0.25)

values = []
for i, j in enumerate(time):
    posVal = np.where(c == j)
    values.append(int(posVal[0]))
val1Pred = []
for i, j in enumerate(values):
    val1Pred.append(a[j])
residuals = val1 - np.array(val1Pred)

from tedi import utils
rms = utils.wrms(residuals, val1err)

print('MAP solution', file=f)
print('kernel:', kernel, file=f)
print('mean:', mean, file=f)
print(file=f)
print('RMS (m/s):', rms, file=f)
print(file=f)

axs[1].set_title('Residuals (RMS = {0}m/s)'.format(np.round(rms, 3)))
axs[1].set_ylabel('FWHM (m/s)')
axs[1].plot(time, residuals, '.k')
axs[1].axhline(y=0, linestyle='--', color='b')
axs[1].set_xlabel('Time (BJD - 2400000)')
plt.savefig('15_residualsPlot.pdf', bbox_inches='tight')
plt.close('all')

f.close()


