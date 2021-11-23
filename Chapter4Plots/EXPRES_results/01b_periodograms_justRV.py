import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.linewidth'] = 2
from astropy.timeseries import LombScargle
linesize = 1

################################################################################
rotP10700 = 34.5
data = np.loadtxt("10700_activity.csv",delimiter=',', skiprows=1, usecols=(1,4,5,11,12))
time1 = data[:,0].T
val1RV, val1RVerr = data[:,1].T, data[:,2].T
val1FW, val1FWerr = data[:,3].T, 2*val1RVerr
val1RVfit = np.poly1d(np.polyfit(time1, val1RV, 1))
val1RV = np.array(val1RV) - val1RVfit(time1)
val1FWfit = np.poly1d(np.polyfit(time1, val1FW, 1))
val1FW = np.array(val1FW) - val1FWfit(time1)

#HD 10700 RVs
f1, p1 = LombScargle(time1, val1RV, val1RVerr).autopower()
bestf = f1[np.argmax(p1)]
bestp = 1/bestf
falseAlarms1 = LombScargle(time1, val1RV, val1RVerr).false_alarm_level([0.1,0.01,0.001])
one = falseAlarms1[1] * np.ones_like(f1)

fig, axs = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(w=7, h=2)
axs[0].errorbar(time1, val1RV, val1RVerr, fmt= '.b', markersize =2, elinewidth=1)
axs[0].tick_params(axis='both', which='both', labelbottom=True)
axs[0].set_ylabel('RVs (m/s)')
axs[0].set_xlabel('Time (MJD)')
axs[1].semilogx(1/f1, p1, color='blue', linewidth=linesize)
axs[1].axvline(x=rotP10700, ymin=p1.min(), ymax=100*p1.max(), color='red', 
                  alpha=1, linewidth=linesize)
axs[1].tick_params(axis='both', which='both', labelbottom=True)
axs[1].plot(1/f1, one, color='red', linestyle='dashed', alpha=1,linewidth=linesize)
axs[1].set_xlim(1,time1.ptp())
axs[1].set_ylabel('Normalized power')
axs[1].set_xlabel('Period (days)')
plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
plt.savefig('periodogram_HD10700.pdf', bbox_inches='tight')
plt.close('all')

################################################################################
rotP26965 = 40
data = np.loadtxt("26965_activity.csv",delimiter=',', skiprows=1, usecols=(1,4,5,11,12))
time2 = data[:,0].T
val2RV, val2RVerr = data[:,1].T, data[:,2].T
val2FW, val2FWerr = data[:,3].T, 2*val2RVerr

val2RVfit = np.poly1d(np.polyfit(time2, val2RV, 1))
val2RV = np.array(val2RV) - val2RVfit(time2)
val2FWfit = np.poly1d(np.polyfit(time2, val2FW, 1))
val2FW = np.array(val2FW) - val2FWfit(time2)

#HD 26965 RVs
f2, p2 = LombScargle(time2, val2RV, val2RVerr).autopower()
bestf = f2[np.argmax(p2)]
bestp = 1/bestf
falseAlarms2 = LombScargle(time2, val2RV, val2RVerr).false_alarm_level([0.1,0.01,0.001])
one = falseAlarms2[1] * np.ones_like(f2)

fig, axs = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(w=7, h=2)
axs[0].errorbar(time2, val2RV, val2RVerr, fmt= '.b', markersize =2, elinewidth=1)
axs[0].tick_params(axis='both', which='both', labelbottom=True)
axs[0].set_ylabel('RVs (m/s)')
axs[0].set_xlabel('Time (MJD)')
axs[1].semilogx(1/f2, p2, color='blue', linewidth=linesize)
axs[1].axvline(x=rotP26965, ymin=p2.min(), ymax=100*p2.max(), color='red', 
                alpha=1, linewidth=linesize)
axs[1].tick_params(axis='both', which='both', labelbottom=True)
axs[1].plot(1/f2, one, color='red', linestyle='dashed', alpha=1, linewidth=linesize)
axs[1].set_xlim(1,time2.ptp())
axs[1].set_ylabel('Normalized power')
axs[1].set_xlabel('Period (days)')
plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
plt.savefig('periodogram_HD26965.pdf', bbox_inches='tight')
plt.close('all')


################################################################################
data = np.loadtxt("34411_activity.csv",delimiter=',', skiprows=1, usecols=(1,4,5,11,12))
time3 = data[:,0].T
val3RV, val3RVerr = data[:,1].T, data[:,2].T
val3FW, val3FWerr = data[:,3].T, 2*val3RVerr

val3RVfit = np.poly1d(np.polyfit(time3, val3RV, 1))
val3RV = np.array(val3RV) - val3RVfit(time3)
val3FWfit = np.poly1d(np.polyfit(time3, val3FW, 1))
val3FW = np.array(val3FW) - val3FWfit(time3)
#HD 34411 RVs
f3, p3 = LombScargle(time3, val3RV, val3RVerr).autopower()
bestf = f3[np.argmax(p3)]
bestp = 1/bestf
falseAlarms3 = LombScargle(time3, val3RV, val3RVerr).false_alarm_level([0.1,0.01,0.001])
one = falseAlarms3[1] * np.ones_like(f3)

fig, axs = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(w=7, h=2)
axs[0].errorbar(time3, val3RV, val3RVerr, fmt= '.b', markersize =2, elinewidth=1)
axs[0].tick_params(axis='both', which='both', labelbottom=True)
axs[0].set_ylabel('RVs (m/s)')
axs[0].set_xlabel('Time (MJD)')
axs[1].semilogx(1/f3, p3, color='blue', linewidth=linesize)
axs[1].tick_params(axis='both', which='both', labelbottom=True)
axs[1].plot(1/f3, one, color='red', linestyle='dashed', alpha=1, linewidth=linesize)
axs[1].set_xlim(1,time3.ptp())
axs[1].set_ylabel('Normalized power')
axs[1].set_xlabel('Period (days)')
plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
plt.savefig('periodogram_HD34411.pdf', bbox_inches='tight')
plt.close('all')

################################################################################
rotP101501 = 17.1
data = np.loadtxt("101501_activity.csv",delimiter=',', skiprows=1, usecols=(1,4,5,11,12))
time4 = data[:,0].T
val4RV, val4RVerr = data[:,1].T, data[:,2].T
val4FW, val4FWerr = data[:,3].T, 2*val4RVerr

val4RVfit = np.poly1d(np.polyfit(time4, val4RV, 1))
val4RV = np.array(val4RV) - val4RVfit(time4)
val4FWfit = np.poly1d(np.polyfit(time4, val4FW, 1))
val4FW = np.array(val4FW) - val4FWfit(time4)
#HD 101501 RVs
f4, p4 = LombScargle(time4, val4RV, val4RVerr).autopower()
bestf = f4[np.argmax(p4)]
bestp = 1/bestf
falseAlarms4 = LombScargle(time4, val4RV, val4RVerr).false_alarm_level([0.1,0.01,0.001])
one = falseAlarms4[1] * np.ones_like(f4)

fig, axs = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(w=7, h=2)
axs[0].errorbar(time4, val4RV, val4RVerr, fmt= '.b', markersize =2, elinewidth=1)
axs[0].tick_params(axis='both', which='both', labelbottom=True)
axs[0].set_ylabel('RVs (m/s)')
axs[0].set_xlabel('Time (MJD)')
axs[1].semilogx(1/f4, p4, color='blue', linewidth=linesize)
axs[1].axvline(x=rotP101501, ymin=p4.min(), ymax=100*p4.max(), color='red', 
               alpha=1, linewidth=linesize)
axs[1].tick_params(axis='both', which='both', labelbottom=True)
axs[1].plot(1/f4, one, color='red', linestyle='dashed', alpha=1, linewidth=linesize)
axs[1].set_xlim(5,time4.ptp())
axs[1].set_ylabel('Normalized power')
axs[1].set_xlabel('Period (days)')
plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
plt.savefig('periodogram_HD101501.pdf', bbox_inches='tight')
plt.close('all')
