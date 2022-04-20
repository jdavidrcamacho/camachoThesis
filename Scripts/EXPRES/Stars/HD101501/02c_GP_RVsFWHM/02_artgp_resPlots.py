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

from artgpn.arttwo import network
from artgpn import weight, node, mean
plt.rcParams['figure.figsize'] = [15, 5]

data = np.loadtxt("101501_activity.csv",delimiter=',', skiprows=1, usecols=(1,4,5,11,12))
time = data[:,0].T
val1, val1err = data[:,1].T, data[:,2].T
val2, val2err = data[:,3].T, 2*val1err

plt.rcParams['figure.figsize'] = [15, 2*5]
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
axs[0].errorbar(time, val1, val1err, fmt='.k')
axs[0].set_ylabel('RV (m/s)')
axs[1].errorbar(time, val2, val2err, fmt='.k')
axs[1].set_xlabel('Time (BJD-2400000)')
axs[1].set_ylabel('FWHM (m/s)')
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

labels = np.array(["$\eta_2$", "$\eta_3$", "$\eta_4$",
                   "$\eta_11$", "$\eta_12$",
                   "slope", "offset", "slope", "offset",
                   "$s_1$", "$s_2$", "logPost"])
corner.corner(combSamples, labels=labels, color="k", bins = 50,
              quantiles=[0.16, 0.5, 0.84], smooth=True, smooth1d=True,
              show_titles=True, plot_density=True, plot_contours=True,
              fill_contours=True, plot_datapoints=True)
plt.savefig('12_cornerPlot_logP.pdf', bbox_inches='tight')
plt.close('all')

labels = np.array(["$\eta_2$", "$\eta_3$", "$\eta_4$",
                   "$\eta_11$", "$\eta_12$",
                   "slope", "offset", "slope", "offset",
                   "$s_1$", "$s_2$"])
corner.corner(combSamples[:,0:-1], labels=labels, color="k", bins = 50,
              quantiles=[0.16, 0.5, 0.84], smooth=True, smooth1d=True,
              show_titles=True, plot_density=True, plot_contours=True,
              fill_contours=True, plot_datapoints=True)
plt.savefig('13_cornerPlot.pdf', bbox_inches='tight')
plt.close('all')

n2, n3,n4,w11,w12,s1,off1,s2,off2,j1,j2,logl = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(combSamples, [16, 50, 84],axis=0)))
print('*** Median values ***', file=f)
print('eta 2= {0[0]} +{0[1]} -{0[2]}'.format(n2), file=f)
print('eta 3= {0[0]} +{0[1]} -{0[2]}'.format(n3), file=f)
print('eta 4= {0[0]} +{0[1]} -{0[2]}'.format(n4), file=f)
print('weight 1= {0[0]} +{0[1]} -{0[2]}'.format(w11), file=f)
print('weight 2= {0[0]} +{0[1]} -{0[2]}'.format(w12), file=f)
print('RV slope = {0[0]} +{0[1]} -{0[2]}'.format(s1), file=f)
print('RV offset = {0[0]} +{0[1]} -{0[2]}'.format(off1), file=f)
print('FWHM slope = {0[0]} +{0[1]} -{0[2]}'.format(s2), file=f)
print('FWHM offset = {0[0]} +{0[1]} -{0[2]}'.format(off2), file=f)
print('RV jitter = {0[0]} +{0[1]} -{0[2]}'.format(j1), file=f)
print('FWHM jitter = {0[0]} +{0[1]} -{0[2]}'.format(j2), file=f)
print('logPost= {0[0]} +{0[1]} -{0[2]}'.format(logl), file=f)
print(file=f)

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

a1, b1, c1 = GPnet.prediction(nodes = nodes, weights = weights, means = means,
                              jitters = jitters,time = tstar, dataset = 1)
bmin1, bmax1 = a1 - b1, a1 + b1
a2, b2, c2 = GPnet.prediction(nodes = nodes, weights = weights, means = means,
                              jitters = jitters,time = tstar, dataset = 2)
bmin2, bmax2 = a2 - b2, a2 + b2

fig, (ax1, ax2) = plt.subplots(nrows=2,ncols=1, sharex=True)
ax1.set_title('Fits')
ax1.fill_between(tstar,bmin1, bmax1, color="red", alpha=0.25)
ax1.plot(tstar, a1, "r-", alpha=1, lw=1.5)
ax1.errorbar(time,val1, val1err, fmt = "k.")
ax1.set_ylabel("RV (m/s)")
ax2.fill_between(tstar,bmin2, bmax2, color="red", alpha=0.25)
ax2.plot(tstar, a2, "r-", alpha=1, lw=1.5)
ax2.errorbar(time,val2, val2err, fmt = "k.")
ax2.set_ylabel("FWHM (m/s)")
plt.savefig('14_mapPlot.pdf', bbox_inches='tight')
plt.close('all')

print('*** MAP values ***', file=f)
print('eta 2= {0}'.format(mapSample[-1,0]), file=f)
print('eta 3= {0}'.format(mapSample[-1,1]), file=f)
print('eta 4= {0}'.format(mapSample[-1,2]), file=f)
print('weight 1= {0}'.format(mapSample[-1,3]**2), file=f)
print('weight 2= {0}'.format(mapSample[-1,4]**2), file=f)
print('RV slope = {0}'.format(mapSample[-1,5]), file=f)
print('RV offset = {0}'.format(mapSample[-1,6]), file=f)
print('FWHM slope = {0}'.format(mapSample[-1,7]), file=f)
print('FWHM offset = {0}'.format(mapSample[-1,8]), file=f)
print('RV jitter = {0}'.format(mapSample[-1,9]), file=f)
print('FWHM jitter = {0}'.format(mapSample[-1,10]), file=f)
print('loglike= {0}'.format(mapSample[-1,11]), file=f)
print(file=f)

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

from artgpn import utils
rms1 = utils.wrms(residuals1, val1err)
rms2 = utils.wrms(residuals2, val2err)

print('RMS RV (m/s):', rms1, file=f)
print('RMS FWHM (m/s):', rms2, file=f)
print(file=f)
f.close()

fig, axs = plt.subplots(2,1, sharex=True, gridspec_kw={'width_ratios': [1],
                                                       'height_ratios': [3, 1]})
axs[0].set_title('Fit')
axs[0].set_ylabel('RV (m/s)')
axs[0].errorbar(time, val1, val1err, fmt= '.k')
axs[0].fill_between(tstar, bmin1, bmax1, color="red", alpha=0.25)
axs[0].plot(tstar, a1, '-r')
axs[1].set_title('Residuals (RMS:{0} m/s)'.format(np.round(rms1, 3)))
axs[1].set_ylabel('RV (m/s)')
axs[1].plot(time, residuals1, '.k')
axs[1].axhline(y=0, linestyle='--', color='b')
axs[1].set_xlabel('Time (BJD - 2400000)')
plt.savefig('15_residualsPlot_RV.pdf', bbox_inches='tight')
plt.close('all')

fig, axs = plt.subplots(2,1, sharex=True, gridspec_kw={'width_ratios': [1],
                                                       'height_ratios': [3, 1]})
axs[0].set_title('Fit')
axs[0].set_ylabel('BIS (m/s)')
axs[0].errorbar(time, val2, val2err, fmt= '.k')
axs[0].plot(tstar, a2, '-r')
axs[0].fill_between(tstar, bmin2, bmax2, color="red", alpha=0.25)
axs[1].set_title('Residuals (RMS:{0})'.format(np.round(rms2, 3)))
axs[1].set_ylabel('FWHM (m/s)')
axs[1].plot(time, residuals2, '.k')
axs[1].axhline(y=0, linestyle='--', color='b')
axs[1].set_xlabel('Time (BJD - 2400000)')
plt.savefig('16_residualsPlot_FWHM.pdf', bbox_inches='tight')
plt.close('all')
