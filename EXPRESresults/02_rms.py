import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')

from gprn.covFunction import SquaredExponential, QuasiPeriodic
from gprn.meanFunction import Linear
from gprn.meanField import inference
from gprn import utils
import emcee 

linesize = 1
storage = 'EXPRES_rms.txt'
f = open(storage, "a")

################################################################################
rotP10700 = 34
data = np.loadtxt("10700_activity.csv",delimiter=',', skiprows=1, usecols=(1,4,5,11,12))
time1 = data[:,0].T
val1RV, val1RVerr = data[:,1].T, data[:,2].T
val1FW, val1FWerr = data[:,3].T, 2*val1RVerr
GPRN1 = inference(1, time1, val1RV, val1RVerr, val1FW, val1FWerr)

filename = "/home/camacho/GPRN/02_EXPRES/New/HD10700/HD10700_RVsFW/savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)
#autocorrelation
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.1 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
samples1 = np.concatenate((samples, log_prob_samples[:, None]), axis=1)
#checking the likelihood that matters to us
values = np.where(samples1[:,-1] == np.max(samples1[:,-1]))
MAPsamples1 = samples1[values,:].reshape(-1, 15)

nodes = [QuasiPeriodic(MAPsamples1[-1,0], MAPsamples1[-1,1], 
                        MAPsamples1[-1,2], MAPsamples1[-1,3])]
weight = [SquaredExponential(MAPsamples1[-1,4], MAPsamples1[-1,5]),
          SquaredExponential(MAPsamples1[-1,6], MAPsamples1[-1,7])]
means = [Linear(MAPsamples1[-1,8], MAPsamples1[-1,9]),
          Linear(MAPsamples1[-1,10], MAPsamples1[-1,11])]
jitter = [MAPsamples1[-1,12], MAPsamples1[-1,13]]

elbo, m, v = GPRN1.ELBOcalc(nodes, weight, means, jitter, 
                            iterations = 50000, mu='init', var='init')
vals, _ = GPRN1.Prediction(nodes, weight, means, jitter, time1, 
                            m, v, variance= True)

print('HD10700 with {0} measurements'.format(time1.size))
print('Timespan = {0} days'.format(time1.ptp()))
print()
rmsHD10700 = utils.rms(val1RV)
print('Initial RMS = {0} m/s'.format(rmsHD10700))
rmsHD10700final = utils.rms(val1RV - vals[0])
print('Final RMS = {0} m/s'.format(rmsHD10700final))
print('RMS reduction = {0}'.format(rmsHD10700/rmsHD10700final))
print()

rmsHD10700 = utils.wrms(val1RV, val1RVerr)
print('Initial weighted RMS = {0} m/s'.format(rmsHD10700))
rmsHD10700final = utils.wrms(val1RV - vals[0], val1RVerr)
print('Final weighted RMS = {0} m/s'.format(rmsHD10700final))
print('weighetd RMS reduction = {0}'.format(rmsHD10700/rmsHD10700final))
print()

f.close()
import sys
sys.exit(0)


################################################################################
rotP26965 = 40
data = np.loadtxt("26965_activity.csv",delimiter=',', skiprows=1, usecols=(1,4,5,11,12))
time2 = data[:,0].T
val2RV, val2RVerr = data[:,1].T, data[:,2].T
val2FW, val2FWerr = data[:,3].T, 2*val2RVerr
GPRN2 = inference(1, time2, val2RV, val2RVerr, val2FW, val2FWerr)

filename = "/home/camacho/GPRN/02_EXPRES/New/HD26965/HD26965_RVsFW/savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)
#autocorrelation
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.1 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
samples2 = np.concatenate((samples, log_prob_samples[:, None]), axis=1)
#checking the likelihood that matters to us
values = np.where(samples2[:,-1] == np.max(samples2[:,-1]))
MAPsamples2 = samples2[values,:].reshape(-1, 15)

nodes = [QuasiPeriodic(MAPsamples2[-1,0], MAPsamples2[-1,1], 
                        MAPsamples2[-1,2], MAPsamples2[-1,3])]
weight = [SquaredExponential(MAPsamples2[-1,4], MAPsamples2[-1,5]),
          SquaredExponential(MAPsamples2[-1,6], MAPsamples2[-1,7])]
means = [Linear(MAPsamples2[-1,8], MAPsamples2[-1,9]),
          Linear(MAPsamples2[-1,10], MAPsamples2[-1,11])]
jitter = [MAPsamples2[-1,12], MAPsamples2[-1,13]]

elbo, m, v = GPRN2.ELBOcalc(nodes, weight, means, jitter, 
                            iterations = 50000, mu='init', var='init')
vals, _ = GPRN2.Prediction(nodes, weight, means, jitter, time2, 
                            m, v, variance= True)

rmsHD26965 = utils.rms(val2RV)
print('HD26965 with {0} measurements'.format(time2.size))
print('Timespan = {0} days'.format(time2.ptp()))
print('Initial RMS = {0} m/s'.format(rmsHD26965))

rmsHD26965final = utils.rms(val2RV - vals[0])
print('Final RMS = {0} m/s'.format(rmsHD26965final))
print('RMS reduction = {0}'.format(rmsHD26965/rmsHD26965final))
print()



################################################################################
data = np.loadtxt("34411_activity.csv",delimiter=',', skiprows=1, usecols=(1,4,5,11,12))
time3 = data[:,0].T
val3RV, val3RVerr = data[:,1].T, data[:,2].T
val3FW, val3FWerr = data[:,3].T, 2*val3RVerr
GPRN3 = inference(1, time3, val3RV, val3RVerr, val3FW, val3FWerr)

filename = "/home/camacho/GPRN/02_EXPRES/New/HD34411/HD34411_RVsFW/savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)
#autocorrelation
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.1 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
samples3 = np.concatenate((samples, log_prob_samples[:, None]), axis=1)
#checking the likelihood that matters to us
values = np.where(samples3[:,-1] == np.max(samples3[:,-1]))
MAPsamples3 = samples3[values,:].reshape(-1, 15)

nodes = [QuasiPeriodic(MAPsamples3[-1,0], MAPsamples3[-1,1], 
                        MAPsamples3[-1,2], MAPsamples3[-1,3])]
weight = [SquaredExponential(MAPsamples3[-1,4], MAPsamples3[-1,5]),
          SquaredExponential(MAPsamples3[-1,6], MAPsamples3[-1,7])]
means = [Linear(MAPsamples3[-1,8], MAPsamples3[-1,9]),
          Linear(MAPsamples3[-1,10], MAPsamples3[-1,11])]
jitter = [MAPsamples3[-1,12], MAPsamples3[-1,13]]

elbo, m, v = GPRN3.ELBOcalc(nodes, weight, means, jitter, 
                            iterations = 50000, mu='init', var='init')
vals, _ = GPRN3.Prediction(nodes, weight, means, jitter, time3, 
                            m, v, variance= True)

rmsHD34411 = utils.rms(val3RV)
print('HD34411 with {0} measurements'.format(time3.size))
print('Timespan = {0} days'.format(time3.ptp()))
print('Initial RMS = {0} m/s'.format(rmsHD34411))

rmsHD34411final = utils.rms(val3RV - vals[0])
print('Final RMS = {0} m/s'.format(rmsHD34411final))
print('RMS reduction = {0}'.format(rmsHD34411/rmsHD34411final))
print()


################################################################################
rotP101501 = 17.1
data = np.loadtxt("101501_activity.csv",delimiter=',', skiprows=1, usecols=(1,4,5,11,12))
time4 = data[:,0].T
val4RV, val4RVerr = data[:,1].T, data[:,2].T
val4FW, val4FWerr = data[:,3].T, 2*val4RVerr
GPRN4 = inference(1, time4, val4RV, val4RVerr, val4FW, val4FWerr)

filename = "/home/camacho/GPRN/02_EXPRES/New/HD101501/HD101501_RVsFW/savedProgress.h5"
sampler = emcee.backends.HDFBackend(filename)
#autocorrelation
tau = sampler.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
thin = int(0.1 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
samples4 = np.concatenate((samples, log_prob_samples[:, None]), axis=1)
#checking the likelihood that matters to us
values = np.where(samples4[:,-1] == np.max(samples4[:,-1]))
MAPsamples4 = samples4[values,:].reshape(-1, 15)

nodes = [QuasiPeriodic(MAPsamples4[-1,0], MAPsamples4[-1,1], 
                       MAPsamples4[-1,2], MAPsamples4[-1,3])]
weight = [SquaredExponential(MAPsamples4[-1,4], MAPsamples4[-1,5]),
          SquaredExponential(MAPsamples4[-1,6], MAPsamples4[-1,7])]
means = [Linear(MAPsamples4[-1,8], MAPsamples4[-1,9]),
         Linear(MAPsamples4[-1,10], MAPsamples4[-1,11])]
jitter = [MAPsamples4[-1,12], MAPsamples4[-1,13]]

elbo, m, v = GPRN4.ELBOcalc(nodes, weight, means, jitter, 
                            iterations = 50000, mu='init', var='init')
vals, _ = GPRN4.Prediction(nodes, weight, means, jitter, time4, 
                           m, v, variance= True)

rmsHD101501 = utils.rms(val4RV)
print('HD101501 with {0} measurements'.format(time4.size))
print('Timespan = {0} days'.format(time4.ptp()))
print('Initial RMS = {0} m/s'.format(rmsHD101501))

rmsHD101501final = utils.rms(val4RV - vals[0])
print('Final RMS = {0} m/s'.format(rmsHD101501final))
print('RMS reduction = {0}'.format(rmsHD101501/rmsHD101501final))
print()

################################################################################
print('HD10700 with {0} measurements'.format(time1.size), file=f)
print('Timespan = {0} days'.format(time1.ptp()), file=f)
print('Initial RMS = {0} m/s'.format(rmsHD10700), file=f)
print('Final RMS = {0} m/s'.format(rmsHD10700final), file=f)
print('RMS reduction = {0}'.format(rmsHD10700/rmsHD10700final), file=f)
print(file=f)
print('HD26965 with {0} measurements'.format(time2.size), file=f)
print('Timespan = {0} days'.format(time2.ptp()), file=f)
print('Initial RMS = {0} m/s'.format(rmsHD26965), file=f)
print('Final RMS = {0} m/s'.format(rmsHD26965final), file=f)
print('RMS reduction = {0}'.format(rmsHD26965/rmsHD26965final), file=f)
print(file=f)
print('HD34411 with {0} measurements'.format(time3.size), file=f)
print('Timespan = {0} days'.format(time3.ptp()), file=f)
print('Initial RMS = {0} m/s'.format(rmsHD34411), file=f)
print('Final RMS = {0} m/s'.format(rmsHD34411final), file=f)
print('RMS reduction = {0}'.format(rmsHD34411/rmsHD34411final), file=f)
print(file=f)
print('HD101501 with {0} measurements'.format(time4.size), file=f)
print('Timespan = {0} days'.format(time4.ptp()), file=f)
print('Initial RMS = {0} m/s'.format(rmsHD101501), file=f)
print('Final RMS = {0} m/s'.format(rmsHD101501final), file=f)
print('RMS reduction = {0}'.format(rmsHD101501/rmsHD101501final), file=f)
print(file=f)
f.close()

print('\n DONE \n')