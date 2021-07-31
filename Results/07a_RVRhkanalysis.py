import numpy as np
import matplotlib
matplotlib.rcParams['figure.figsize'] = [7, 3]
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')

import emcee

from artgpn.arttwo import network
import artgpn as art
import gprn as gprn

time,rv,rverr,rhk,rhkerr= np.loadtxt("/home/camacho/Github/camachoThesis/sunData/sunBinned_Dumusque.txt", 
                                     skiprows = 1,  unpack = True, usecols = (0,1,2,3,4))
time, val1, val1err = time, rv, rverr
val2, val2err = rhk, rhkerr

gpResults = "/home/camacho/GPRN/01_SUN/70_sun/GP/07b_GP_RVsRhk/savedProgress.h5"
gprnResults = "/home/camacho/GPRN/01_SUN/70_sun/GPRN/07a_gprn_RVsRhk/savedProgress.h5"

gplabels = np.array(["$\eta_2$", "$\eta_3$", "$\eta_4$", "$\eta_11$", "$\eta_12$",
                   "slope", "offset", "slope", "offset", "$s_1$", "$s_2$"])

gprnlabels = np.array(["eta1", "eta2", "eta3", "eta4", "eta11", "eta21","eta12", "eta22",
                       "slope1", "offset1","slope2", "offset2", "jitter1","jitter2"])

gpsampler = emcee.backends.HDFBackend(gpResults)
gptau = gpsampler.get_autocorr_time(tol=0)
gpburnin = int(2*np.max(gptau))
gpthin = int(0.1 * np.min(gptau))
gpsamples = gpsampler.get_chain(discard=gpburnin, flat=True, thin=gpthin)
log_prob_samples = gpsampler.get_log_prob(discard=gpburnin, flat=True, thin=gpthin)
gpCombSamples = np.concatenate((gpsamples, log_prob_samples[:, None]), axis=1)

gprnsampler = emcee.backends.HDFBackend(gprnResults)
gprntau = gprnsampler.get_autocorr_time(tol=0)
gprnburnin = int(2*np.max(gprntau))
gprnthin = int(0.1 * np.min(gprntau))
gprnsamples = gprnsampler.get_chain(discard=gprnburnin, flat=True, thin=gprnthin)
log_prob_samples = gprnsampler.get_log_prob(discard=gprnburnin, flat=True, thin=gprnthin)
gprnCombSamples = np.concatenate((gprnsamples, log_prob_samples[:, None]), axis=1)

##### GP stuff
values = np.where(gpCombSamples[:,-1] == np.max(gpCombSamples[:,-1]))
gpMapSample = gpCombSamples[values,:].reshape(-1, 12)

#Having a solution from our MCMC we need to redefine all the network
nodes = [art.node.QuasiPeriodic(gpMapSample[-1,0], gpMapSample[-1,1], gpMapSample[-1,2])]
weights = [art.weight.Constant(gpMapSample[-1,3]),
           art.weight.Constant(gpMapSample[-1,4])]
means = [art.mean.Linear(gpMapSample[-1,5], gpMapSample[-1,6]),
         art.mean.Linear(gpMapSample[-1,7], gpMapSample[-1,8])]
jitters = [gpMapSample[-1,9], gpMapSample[-1,10]]

GPnet = network(1, time, val1, val1err, val2, val2err)
tstar = np.linspace(time.min()-1, time.max()+1, 10000)

mu11, std11, cov11 = GPnet.prediction(nodes = nodes, weights = weights, means = means,
                                      jitters = jitters,time = tstar, dataset = 1)
mu22, std22, cov22 = GPnet.prediction(nodes = nodes, weights = weights, means = means,
                                      jitters = jitters, time = tstar, dataset = 2)
fig, axs = plt.subplots(nrows=4,ncols=1, sharex=True,
                        gridspec_kw={'width_ratios': [1],
                                     'height_ratios': [2.25, 1, 2.25, 1]})
fig.set_size_inches(w=7, h=4+4)
axs[0].fill_between(tstar,mu11+std11, mu11-std11, color="red", alpha=0.25)
axs[0].plot(tstar, mu11, '-r', alpha=0.75, label='GP')
axs[0].errorbar(time,val1, val1err, fmt = "k.")
axs[0].set_ylabel("RV (m/s)")
axs[2].fill_between(tstar, mu22+std22, mu22-std22, color="red", alpha=0.25)
axs[2].plot(tstar, mu22, '-r', alpha=0.75, label='GP')
axs[2].errorbar(time,val2, val2err, fmt = "k.")
axs[2].set_ylabel("log Rhk")

vals1, _, _ = GPnet.prediction(nodes = nodes, weights = weights, means = means,
                              jitters = jitters,time = time, dataset = 1)
gpresiduals1 = val1 - vals1
vals2, _, _ = GPnet.prediction(nodes = nodes, weights = weights, means = means,
                              jitters = jitters,time = time, dataset = 2)
gpresiduals2 = val2 - vals2

axs[1].axhline(y=0, linestyle='--', color='k')
axs[1].plot(time, gpresiduals1, '*r', alpha=1, label='GP')
axs[1].set_ylabel('Residuals (m/s)')
axs[3].axhline(y=0, linestyle='--', color='k')
axs[3].plot(time, gpresiduals2, '*r', alpha=1, label='GP')
axs[3].set_ylabel('Residuals (m/s)')

##### GPRN stuff
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

a, b = GPRN.Prediction(nodes, weight, means, jitter, tstar, vm, vv, variance= True)
bmin0, bmax0 = a[0]-np.sqrt(b[0]), a[0]+np.sqrt(b[0])
bmin1, bmax1 = a[1]-np.sqrt(b[1]), a[1]+np.sqrt(b[1])

axs[0].plot(tstar, a[0].T, '--b', alpha=0.75, label='GPRN')
axs[0].fill_between(tstar,  bmax0.T, bmin0.T, color="blue", alpha=0.25)
axs[0].legend(loc='upper right', facecolor='white', framealpha=1, edgecolor='black')

axs[2].plot(tstar, a[1].T, '--b', alpha=0.75, label='GPRN')
axs[2].fill_between(tstar,  bmax1.T, bmin1.T, color="blue", alpha=0.25)
axs[2].legend(loc='upper right', facecolor='white', framealpha=1, edgecolor='black')
axs[2].set_xlabel('Time (BJD - 2400000)')


vals,_= GPRN.Prediction(nodes,weight, means, jitter, time, vm, vv)
gprnresiduals1 = val1 - vals[0]
gprnresiduals2 = val2 - vals[1]
axs[1].axhline(y=0, linestyle='--', color='k')
axs[1].plot(time, gprnresiduals1, '.b', mfc='none', alpha=1, label='GPRN')
axs[1].set_ylabel('Residuals (m/s)')
axs[1].legend(loc='upper left', bbox_to_anchor=(0, 0.8), 
              facecolor='white', framealpha=1, edgecolor='black')
axs[3].axhline(y=0, linestyle='--', color='k')
axs[3].plot(time, gprnresiduals2, '.b', mfc='none', alpha=1, label='GPRN')
axs[3].set_ylabel('Residuals (m/s)')
axs[3].legend(loc='upper left', bbox_to_anchor=(0, 0.8), 
              facecolor='white', framealpha=1, edgecolor='black')

plt.tight_layout()
plt.savefig('RVRhkfit_withResidualsPlot.pdf', bbox_inches='tight')
# plt.close('all')


# ard = np.abs(gprnresiduals) - np.abs(gpresiduals)
# plt.figure()
# plt.title('Residual difference (GPRN - GP)')
# plt.plot(time, ard, '.b')
# plt.ylabel('Difference (m/s)')
# plt.xlabel('Time (BJD - 2400000)')
# plt.axhline(y=0, linestyle='--', color='k')
# plt.tight_layout()
# plt.savefig('RVfit_ResidualDifferencePlot.pdf', bbox_inches='tight')
# plt.close('all')

# total = 0
# totals = [0]
# for i, j in enumerate(ard):
#     if j < 0:
#         total += 1
#         totals.append(total / (i+1))
#         print(total, 'out of', i+1, 'then', total / (i+1))
#     else:
#         totals.append(totals[-1])
# plt.figure()
# plt.title('GPRN best RMS percentage')
# plt.plot(totals[1:], '.-b')
# plt.xlabel('Number of points')
# plt.ylabel('%')
# plt.savefig('RVfit_percentagePlot.pdf', bbox_inches='tight')
# plt.close('all')
