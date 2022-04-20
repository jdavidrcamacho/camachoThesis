import numpy as np
import matplotlib
matplotlib.rcParams['figure.figsize'] = [4, 4]
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.linewidth'] = 2
import emcee
import gprn as gprn

plt.figure()
plt.xlabel('Radial velocity (m/s)')
plt.ylabel('Frequency')
time,rv,rverr,bis,biserr= np.loadtxt("/home/camacho/GitHub/camachoThesis/Chapter4Plots/SUN_periodograms/sunBinned_Dumusque.txt", 
                                     skiprows = 1,  unpack = True, usecols = (0,1,2,7,8))
time, val1, val1err = time, rv, rverr
val2, val2err = bis, biserr
Results_RVsBis = "/media/camacho/HDD 1 tb/GPRN/01_SUN/70_sun/GPRN/05a_gprn_RVsBIS/savedProgress.h5"
gprnsampler = emcee.backends.HDFBackend(Results_RVsBis)
gprntau = gprnsampler.get_autocorr_time(tol=0)
gprnburnin = int(2*np.max(gprntau))
gprnthin = int(0.1 * np.min(gprntau))
gprnsamples = gprnsampler.get_chain(discard=gprnburnin, flat=True, thin=gprnthin)
log_prob_samples = gprnsampler.get_log_prob(discard=gprnburnin, flat=True, thin=gprnthin)
gprnCombSamples = np.concatenate((gprnsamples, log_prob_samples[:, None]), axis=1)
values = np.where(gprnCombSamples[:,-1] == np.max(gprnCombSamples[:,-1]))
gprnMapSample = gprnCombSamples[values,:].reshape(-1, 15)

nodes = [gprn.covFunction.QuasiPeriodic(gprnMapSample[-1,0], gprnMapSample[-1,1], 
                                        gprnMapSample[-1,2], gprnMapSample[-1,3])]
weight = [gprn.covFunction.SquaredExponential(gprnMapSample[-1,4], gprnMapSample[-1,5]),
          gprn.covFunction.SquaredExponential(gprnMapSample[-1,6], gprnMapSample[-1,7])]
means = [gprn.meanFunction.Linear(gprnMapSample[-1,8], gprnMapSample[-1,9]),
         gprn.meanFunction.Linear(gprnMapSample[-1,10], gprnMapSample[-1,11])]
jitter = [gprnMapSample[-1,12], gprnMapSample[-1,13]]
GPRN = gprn.meanField.inference(1, time, val1, val1err, val2, val2err)
elbo, vm, vv = GPRN.ELBOcalc(nodes, weight, means, jitter, iterations = 5000,
                             mu='init', var='init')
elbo, m, v = GPRN.ELBOcalc(nodes, weight, means, jitter, 
                              iterations = 50000, mu='init', var='init')
tstar = np.linspace(time.min()-10, time.max()+10, 2500)
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
gprnresiduals1 = val1 - np.array(val1Pred)
gprnresiduals2 = val2 - np.array(val2Pred)
from gpyrn import _utils
rms1 = _utils.wrms(gprnresiduals1, val1err)
rms2 = _utils.wrms(gprnresiduals2, val2err)

plt.hist(gprnresiduals1, density=True, color="blue",alpha=0.5, label ="$\sigma$ = 1.176 m/s")

###############################################################################
time,rv,rverr,bis,biserr= np.loadtxt("/home/camacho/GitHub/camachoThesis/Chapter4Plots/SUN_periodograms/sunBinned_Dumusque.txt", 
                                     skiprows = 1,  unpack = True, usecols = (0,1,2,9,10))
time, val1, val1err = time, rv, rverr
val2, val2err = bis, biserr
Results_RVsBis = "/media/camacho/HDD 1 tb/GPRN/01_SUN/70_sun/GPRN/06a_gprn_RVsFW/savedProgress.h5"
gprnsampler = emcee.backends.HDFBackend(Results_RVsBis)
gprntau = gprnsampler.get_autocorr_time(tol=0)
gprnburnin = int(2*np.max(gprntau))
gprnthin = int(0.1 * np.min(gprntau))
gprnsamples = gprnsampler.get_chain(discard=gprnburnin, flat=True, thin=gprnthin)
log_prob_samples = gprnsampler.get_log_prob(discard=gprnburnin, flat=True, thin=gprnthin)
gprnCombSamples = np.concatenate((gprnsamples, log_prob_samples[:, None]), axis=1)
values = np.where(gprnCombSamples[:,-1] == np.max(gprnCombSamples[:,-1]))
gprnMapSample = gprnCombSamples[values,:].reshape(-1, 15)

nodes = [gprn.covFunction.QuasiPeriodic(gprnMapSample[-1,0], gprnMapSample[-1,1], 
                                        gprnMapSample[-1,2], gprnMapSample[-1,3])]
weight = [gprn.covFunction.SquaredExponential(gprnMapSample[-1,4], gprnMapSample[-1,5]),
          gprn.covFunction.SquaredExponential(gprnMapSample[-1,6], gprnMapSample[-1,7])]
means = [gprn.meanFunction.Linear(gprnMapSample[-1,8], gprnMapSample[-1,9]),
         gprn.meanFunction.Linear(gprnMapSample[-1,10], gprnMapSample[-1,11])]
jitter = [gprnMapSample[-1,12], gprnMapSample[-1,13]]
GPRN = gprn.meanField.inference(1, time, val1, val1err, val2, val2err)
elbo, vm, vv = GPRN.ELBOcalc(nodes, weight, means, jitter, iterations = 5000,
                             mu='init', var='init')
elbo, m, v = GPRN.ELBOcalc(nodes, weight, means, jitter, 
                              iterations = 50000, mu='init', var='init')
tstar = np.linspace(time.min()-10, time.max()+10, 2500)
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
gprnresiduals1 = val1 - np.array(val1Pred)
gprnresiduals2 = val2 - np.array(val2Pred)
from gpyrn import _utils
rms1 = _utils.wrms(gprnresiduals1, val1err)
rms2 = _utils.wrms(gprnresiduals2, val2err)

plt.hist(gprnresiduals1, density=True, color ="red", alpha=0.5, label ="$\sigma$ = 1.034 m/s")

###############################################################################
time,rv,rverr,bis,biserr= np.loadtxt("/home/camacho/GitHub/camachoThesis/Chapter4Plots/SUN_periodograms/sunBinned_Dumusque.txt", 
                                     skiprows = 1,  unpack = True, usecols = (0,1,2,3,4))
time, val1, val1err = time, rv, rverr
val2, val2err = bis, biserr
Results_RVsBis = "/media/camacho/HDD 1 tb/GPRN/01_SUN/70_sun/GPRN/07a_gprn_RVsRhk/savedProgress.h5"
gprnsampler = emcee.backends.HDFBackend(Results_RVsBis)
gprntau = gprnsampler.get_autocorr_time(tol=0)
gprnburnin = int(2*np.max(gprntau))
gprnthin = int(0.1 * np.min(gprntau))
gprnsamples = gprnsampler.get_chain(discard=gprnburnin, flat=True, thin=gprnthin)
log_prob_samples = gprnsampler.get_log_prob(discard=gprnburnin, flat=True, thin=gprnthin)
gprnCombSamples = np.concatenate((gprnsamples, log_prob_samples[:, None]), axis=1)
values = np.where(gprnCombSamples[:,-1] == np.max(gprnCombSamples[:,-1]))
gprnMapSample = gprnCombSamples[values,:].reshape(-1, 15)

nodes = [gprn.covFunction.QuasiPeriodic(gprnMapSample[-1,0], gprnMapSample[-1,1], 
                                        gprnMapSample[-1,2], gprnMapSample[-1,3])]
weight = [gprn.covFunction.SquaredExponential(gprnMapSample[-1,4], gprnMapSample[-1,5]),
          gprn.covFunction.SquaredExponential(gprnMapSample[-1,6], gprnMapSample[-1,7])]
means = [gprn.meanFunction.Linear(gprnMapSample[-1,8], gprnMapSample[-1,9]),
         gprn.meanFunction.Linear(gprnMapSample[-1,10], gprnMapSample[-1,11])]
jitter = [gprnMapSample[-1,12], gprnMapSample[-1,13]]
GPRN = gprn.meanField.inference(1, time, val1, val1err, val2, val2err)
elbo, vm, vv = GPRN.ELBOcalc(nodes, weight, means, jitter, iterations = 5000,
                             mu='init', var='init')
elbo, m, v = GPRN.ELBOcalc(nodes, weight, means, jitter, 
                              iterations = 50000, mu='init', var='init')
tstar = np.linspace(time.min()-10, time.max()+10, 2500)
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
gprnresiduals1 = val1 - np.array(val1Pred)
gprnresiduals2 = val2 - np.array(val2Pred)
from gpyrn import _utils
rms1 = _utils.wrms(gprnresiduals1, val1err)
rms2 = _utils.wrms(gprnresiduals2, val2err)

plt.hist(gprnresiduals1, density=True, color ="green", alpha=0.5, label ="$\sigma$ = 1.103 m/s")
plt.legend(facecolor='whitesmoke', framealpha=1, edgecolor='black')

from matplotlib.ticker import AutoMinorLocator
plt.grid(which='major', alpha=0.5)
plt.grid(which='minor', alpha=0.2)

plt.tight_layout(pad=0.1, h_pad=1, w_pad=0.1)
plt.savefig('rmsHist.pdf', bbox_inches='tight')
plt.show()