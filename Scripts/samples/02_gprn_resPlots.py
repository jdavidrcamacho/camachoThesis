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
from matplotlib.ticker import AutoMinorLocator
plt.close('all')

from gpyrn import meanfield, covfunc, meanfunc

import emcee

time,rv,rverr,bis,biserr,fw, fwerr = np.loadtxt("sample50points.txt",
                                                skiprows = 1, unpack = True, 
                                                usecols = (0,1,2,3,4,5,6))
val1, val1err = rv, rverr
val2,val2err = bis, biserr
val3,val3err = fw, fwerr
GPRN = meanfield.inference(1, time, val1,val1err, val3,val3err)

plt.rcParams['figure.figsize'] = [7, 3]
plt.errorbar(time,val1,val1err, fmt='.k', label='data')
plt.legend()
plt.title('Data')
plt.xlabel('Time (days)')
plt.ylabel('RV (m/s)')
plt.savefig('new11_datasetA.pdf', bbox_inches='tight')
plt.close('all')

plt.errorbar(time,val3,val3err, fmt='.k', label='data')
plt.legend()
plt.title('Data')
plt.xlabel('Time (days)')
plt.ylabel('BIS (m/s)')
plt.savefig('new11_datasetC.pdf', bbox_inches='tight')
plt.close('all')

plt.rcParams['figure.figsize'] = [7, 7]

filename =  "savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)
print(sampler.iteration)

storage_name = "10_results.txt"
f = open(storage_name, "a")
print('iterations:', sampler.iteration, file=f)

#autocorrelation
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.5 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)

print("burn-in: {0}".format(burnin), file=f)
print("thin: {0}".format(thin), file=f)
print("flat chain shape: {0}".format(samples.shape), file=f)
print(file=f)
combSamples = np.concatenate((samples, log_prob_samples[:, None]), axis=1)



#plotting corner
import corner

labels = np.array(["$\eta_2^{n}$", "$\eta_3^{n}$", "$\eta_4^{n}$",
                   "$\eta_1^{w1}$", "$\eta_2^{w1}$",
                   "$\eta_1^{w3}$", "$\eta_2^{w3}$"])
                   
truths=[17, 23, 0.75, 7,29, 7,109]
                   
# corner.corner(combSamples [:,0:-4], truths=truths,  labels=labels, color="k", bins = 50,
#               quantiles=[0.16, 0.5, 0.84], smooth=True, smooth1d=True, 
#               show_titles=True, plot_density=True, plot_contours=True,
#               fill_contours=True, plot_datapoints=True)
# plt.savefig('new12_cornerPlot.pdf', bbox_inches='tight')
# plt.close('all')

corner.corner(combSamples, color="k", bins = 50,
              quantiles=[0.16, 0.5, 0.84], smooth=True, smooth1d=True, 
              show_titles=True, plot_density=True, plot_contours=True,
              fill_contours=True, plot_datapoints=True)
plt.savefig('new12b_cornerPlot.pdf', bbox_inches='tight')
plt.close('all')


n2,n3,n4,w11,w21,w13,w23,j1,j3,elbo = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
                            zip(*np.percentile(combSamples, [16, 50, 84],axis=0)))
print(file=f)
print('*** Median values ***', file=f)
print('node eta2= {0[0]} +{0[1]} -{0[2]}'.format(n2), file=f)
print('node eta3= {0[0]} +{0[1]} -{0[2]}'.format(n3), file=f)
print('node eta4= {0[0]} +{0[1]} -{0[2]}'.format(n4), file=f)
print('weight1 eta1= {0[0]} +{0[1]} -{0[2]}'.format(w11), file=f)
print('weight1 eta2= {0[0]} +{0[1]} -{0[2]}'.format(w21), file=f)
print('weight3 eta1= {0[0]} +{0[1]} -{0[2]}'.format(w13), file=f)
print('weight3 eta2= {0[0]} +{0[1]} -{0[2]}'.format(w23), file=f)
print('jitter1= {0[0]} +{0[1]} -{0[2]}'.format(j1), file=f)
print('jitter3= {0[0]} +{0[1]} -{0[2]}'.format(j3), file=f)
print('logPost= {0[0]} +{0[1]} -{0[2]}'.format(elbo), file=f)
print(file=f)

#checking the logPost that matters to us
values = np.where(combSamples[:,-1] == np.max(combSamples[:,-1]))
mapSample = combSamples[values,:]
mapSample = mapSample.reshape(-1, 10)
nodes = [covfunc.QuasiPeriodic(1,mapSample[-1,0], mapSample[-1,1], mapSample[-1,2])]
weight = [covfunc.SquaredExponential(mapSample[-1,3], mapSample[-1,4]),
          covfunc.SquaredExponential(mapSample[-1,5], mapSample[-1,6])]
means = [meanfunc.Constant(0),  meanfunc.Constant(0)]
jitter = [mapSample[-1,7], mapSample[-1,8]]

elbo, m, v = GPRN.ELBOcalc(nodes, weight, means, jitter, iterations = 50000,
                           mu='init', var='init')
    
tstar = np.linspace(time.min()-10, time.max()+10, 2500)
tstar = np.sort(np.concatenate((tstar, time)))
a, b, c = GPRN.Prediction(nodes, weight, means, jitter, tstar, m, v)
bmin, bmax = a-np.sqrt(b), a+np.sqrt(b)

print('*** MAP values ***', file=f)
print('node eta2= {0}'.format(mapSample[-1,0]), file=f)
print('node eta3= {0}'.format(mapSample[-1,1]), file=f)
print('node eta4= {0}'.format(mapSample[-1,2]), file=f)
print('weight1 eta1= {0}'.format(mapSample[-1,3]), file=f)
print('weight1 eta2= {0}'.format(mapSample[-1,4]), file=f)
print('weight1 eta1= {0}'.format(mapSample[-1,5]), file=f)
print('weight1 eta2= {0}'.format(mapSample[-1,6]), file=f)
print('offset= {0}'.format(mapSample[-1,7]), file=f)
print('jitter= {0}'.format(mapSample[-1,8]), file=f)
print('logPost= {0}'.format(mapSample[-1,9]), file=f)
print(file=f)
print('done')

plt.rcParams['figure.figsize'] = [7, 10]
fig, axs = plt.subplots(4,1, sharex=True, gridspec_kw={'width_ratios': [1],
                                                       'height_ratios': [3,1,3,1]})
axs[0].set_ylabel('RV (m/s)')
axs[0].errorbar(time, val1, val1err, fmt= '.k')
axs[0].plot(tstar, a[:,0].T, '-r')
axs[0].fill_between(tstar,  bmax[:,0].T, bmin[:,0].T, color="red", alpha=0.25)
values = []
for i, j in enumerate(time):
    posVal = np.where(c == j)
    values.append(int(posVal[0]))
val1Pred = []
for i, j in enumerate(values):
    val1Pred.append(a[j,0])
residuals = val1 - np.array(val1Pred)

from gprn import utils
rms = utils.wrms(residuals, val1err)
print('RMS (m/s):', rms, file=f)

axs[1].set_ylabel('Residuals')
axs[1].errorbar(time, residuals, val1err, fmt= '.k')
axs[1].axhline(y=0, linestyle='--', color='b')
axs[1].set_xlabel('Time (BJD - 2400000)')
axs[0].xaxis.set_minor_locator(AutoMinorLocator(5))
axs[0].yaxis.set_minor_locator(AutoMinorLocator(5))
axs[0].grid(which='major', alpha=0.5)
axs[0].grid(which='minor', alpha=0.2)
axs[1].xaxis.set_minor_locator(AutoMinorLocator(5))
axs[1].yaxis.set_minor_locator(AutoMinorLocator(5))
axs[1].grid(which='major', alpha=0.5)
axs[1].grid(which='minor', alpha=0.2)


axs[2].set_ylabel('FWHM (m/s)')
axs[2].errorbar(time, val3, val3err, fmt= '.k')
axs[2].plot(tstar, a[:,1].T, '-r')
axs[2].fill_between(tstar,  bmax[:,1].T, bmin[:,1].T, color="red", alpha=0.25)
values = []
for i, j in enumerate(time):
    posVal = np.where(c == j)
    values.append(int(posVal[0]))
val3Pred = []
for i, j in enumerate(values):
    val3Pred.append(a[j,1])
residuals = val3 - np.array(val3Pred)

from gprn import utils
rms = utils.wrms(residuals, val3err)
print('RMS (m/s):', rms, file=f)

axs[3].set_ylabel('Residuals')
axs[3].errorbar(time, residuals, val3err, fmt= '.k')
axs[3].axhline(y=0, linestyle='--', color='b')
axs[3].set_xlabel('Time (BJD - 2400000)')
axs[2].xaxis.set_minor_locator(AutoMinorLocator(5))
axs[2].yaxis.set_minor_locator(AutoMinorLocator(5))
axs[2].grid(which='major', alpha=0.5)
axs[2].grid(which='minor', alpha=0.2)
axs[3].xaxis.set_minor_locator(AutoMinorLocator(5))
axs[3].yaxis.set_minor_locator(AutoMinorLocator(5))
axs[3].grid(which='major', alpha=0.5)
axs[3].grid(which='minor', alpha=0.2)

plt.savefig('new13_plotWithResiduals.pdf', bbox_inches='tight')
plt.close('all')
print(file=f)
f.close()


