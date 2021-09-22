import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator
plt.close('all')

###### Data .rdb file #####
time,rv,rverr,rhk,rhkerr,bis,biserr,fw,fwerr = np.loadtxt("/home/camacho/GPRN/Data/sunBinned_Dumusque.txt", 
                                                          skiprows = 1, 
                                                          unpack = True, 
                                                          usecols = (0,1,2,3,4,7,8,9,10))
val1, val1err = rv, rverr
val2, val2err = bis, biserr
val3, val3err = fw, fwerr
val4, val4err = rhk, rhkerr

from scipy import signal

val11 = val1 - np.mean(val1)
corr0 = signal.correlate(val11, val11, mode='full', method='direct')
lags0 = signal.correlation_lags(len(val11), len(val1), mode='full')
corr0 = corr0 / (val11.size * val1.std() * val1.std())
# corr0 /= np.max(corr0)

val22 = val2 - np.mean(val2)
corr1 = signal.correlate(val11, val22, mode='full', method='direct')
lags1 = signal.correlation_lags(len(val11), len(val22), mode='full')
corr1 = corr1 / (val11.size * val1.std() * val2.std())
#corr1 /= np.max(corr1)

val33 = val3 - np.mean(val3)
corr2 = signal.correlate(val11, val33, mode='full', method='direct')
lags2 = signal.correlation_lags(len(val11), len(val33), mode='full')
corr2 = corr2 / (val11.size * val1.std() * val3.std())
#corr2 /= np.max(corr2)

val44 = val4 - np.mean(val4)
corr3 = signal.correlate(val11, val44, mode='full', method='direct')
lags3 = signal.correlation_lags(len(val11), len(val44), mode='full')
corr3 = corr3 / (val11.size * val1.std() * val4.std())
#corr3 /= np.max(corr3)

plt.rcParams['figure.figsize'] = [7, 5]
fig, axs = plt.subplots(2, 1)
axs[0].xaxis.set_minor_locator(AutoMinorLocator(5))
axs[0].yaxis.set_minor_locator(AutoMinorLocator(5))
axs[0].grid(which='major', alpha=0.5)
axs[0].grid(which='minor', alpha=0.2)

# axs[0].plot(lags0, corr0, '-k', label='RV and RV')
axs[0].plot(lags1, corr1, '-.g', label='RV and BIS')
axs[0].plot(lags2, corr2, '--r', label='RV and FWHM')
axs[0].plot(lags3, corr3, ':b', label='RV and $\log R^{\'}_{hk}$')
axs[0].set_xlim(-15, 15)
axs[0].set_ylabel('Scipy')
axs[0].set_xlabel('')
axs[0].tick_params(axis='both', which='both', labelbottom=True)
axs[0].legend(loc='lower left', facecolor='white', framealpha=1, edgecolor='black')
# plt.close('all')

from correlation_functions import DCF_EK

axs[1].xaxis.set_minor_locator(AutoMinorLocator(5))
axs[1].yaxis.set_minor_locator(AutoMinorLocator(5))
axs[1].grid(which='major', alpha=0.5)
axs[1].grid(which='minor', alpha=0.2)

EKbins = np.linspace(-15, 15, 31)
C_EK, C_EK_err, bins = DCF_EK(time, val11, val11, val1err, val1err, bins=EKbins)
t_EK = 0.5 * (bins[1:] + bins[:-1])
m = ~np.isnan(C_EK)
# axs[1].plot(t_EK[m], C_EK[m],  '-k', label='RV and RV')

EKbins = np.linspace(-15, 15, 31)
C_EK, C_EK_err, bins = DCF_EK(time, val11, val22, val1err, val2err, bins=EKbins)
t_EK = 0.5 * (bins[1:] + bins[:-1])
m = ~np.isnan(C_EK)
axs[1].plot(t_EK[m], C_EK[m],  '-.g', label='RV and BIS')

EKbins = np.linspace(-15, 15, 31)
C_EK, C_EK_err, bins = DCF_EK(time, val11, val33, val1err, val3err, bins=EKbins)
t_EK = 0.5 * (bins[1:] + bins[:-1])
m = ~np.isnan(C_EK)
axs[1].plot(t_EK[m], C_EK[m], '--r', label='RV and FWHM')

EKbins = np.linspace(-15, 15, 31)
C_EK, C_EK_err, bins = DCF_EK(time, val11, val44, val1err, val4err, bins=EKbins)
t_EK = 0.5 * (bins[1:] + bins[:-1])
m = ~np.isnan(C_EK)
axs[1].plot(t_EK[m], C_EK[m], ':b', label='RV and $\log R^{\'}_{hk}$')
axs[1].legend(loc='lower left', facecolor='white', framealpha=1, edgecolor='black')
axs[1].set_xlim(-15, 15)
axs[1].grid(True)
axs[1].set_ylabel('Edelson-Krolik method')
axs[1].set_xlabel('Lag (days)')
plt.savefig('lags.pdf', bbox_inches='tight')
# plt.close('all')

#################################################################################
time,rv,rverr,bis,biserr,fw,fwerr = np.loadtxt("/home/camacho/GPRN/Data/sunBinned_Cameron.txt", 
                                                          skiprows = 1, unpack = True, 
                                                          usecols = (0,1,2,3,4,5,6))
val1, val1err = rv, rverr
val2, val2err = bis, biserr
val3, val3err = fw, fwerr

val11 = val1 - np.mean(val1)
val22 = val2 - np.mean(val2)
val33 = val3 - np.mean(val3)

corr0 = signal.correlate(val11, val11, mode='full', method='direct')
lags0 = signal.correlation_lags(len(val11), len(val1), mode='full')
corr0 = corr0 / (val11.size * val1.std() * val1.std())
# corr0 /= np.max(corr0)

corr1 = signal.correlate(val11, val22, mode='full', method='direct')
lags1 = signal.correlation_lags(len(val11), len(val22), mode='full')
corr1 = corr1 / (val11.size * val1.std() * val2.std())
#corr1 /= np.max(corr1)

corr2 = signal.correlate(val11, val33, mode='full', method='direct')
lags2 = signal.correlation_lags(len(val11), len(val33), mode='full')
corr2 = corr2 / (val11.size * val1.std() * val3.std())
#corr2 /= np.max(corr2)

plt.rcParams['figure.figsize'] = [7, 5]
fig, axs = plt.subplots(2, 1)
axs[0].xaxis.set_minor_locator(AutoMinorLocator(5))
axs[0].yaxis.set_minor_locator(AutoMinorLocator(5))
axs[0].grid(which='major', alpha=0.5)
axs[0].grid(which='minor', alpha=0.2)

# axs[0].plot(lags0, corr0, '-k', label='RV and RV')
axs[0].plot(lags1, corr1, '-.r', label='RV and BIS')
axs[0].plot(lags2, corr2, '--b', label='RV and FWHM')
axs[0].axvline(x=0, linestyle='--', color='gray')
axs[0].set_xlim(-15, 15)
axs[0].set_ylabel('Scipy')
axs[0].set_xlabel('')
axs[0].tick_params(axis='both', which='both', labelbottom=True)
axs[0].legend(loc='lower left', facecolor='white', framealpha=1, edgecolor='black')

from correlation_functions import DCF_EK

axs[1].xaxis.set_minor_locator(AutoMinorLocator(5))
axs[1].yaxis.set_minor_locator(AutoMinorLocator(5))
axs[1].grid(which='major', alpha=0.5)
axs[1].grid(which='minor', alpha=0.2)
axs[1].set_ylabel('Edelson-Krolik method')

EKbins = np.linspace(-15, 15, 31)
C_EK, C_EK_err, bins = DCF_EK(time, val11, val11, val1err, val1err, bins=EKbins)
t_EK = 0.5 * (bins[1:] + bins[:-1])
m = ~np.isnan(C_EK)
#axs[1].plot(t_EK[m], C_EK[m],  '-k', label='RV and RV')

EKbins = np.linspace(-15, 15, 31)
C_EK, C_EK_err, bins = DCF_EK(time, val1, val22, val1err, val2err, bins=EKbins)
t_EK = 0.5 * (bins[1:] + bins[:-1])
m = ~np.isnan(C_EK)
axs[1].plot(t_EK[m], C_EK[m],  '-.r', label='RV and BIS')

EKbins = np.linspace(-15, 15, 31)
C_EK, C_EK_err, bins = DCF_EK(time, val11, val33, val1err, val3err, bins=EKbins)
t_EK = 0.5 * (bins[1:] + bins[:-1])
m = ~np.isnan(C_EK)
axs[1].plot(t_EK[m], C_EK[m], '--b', label='RV and FWHM')

axs[1].legend(loc='lower left', facecolor='white', framealpha=1, edgecolor='black')
axs[1].axvline(x=0, linestyle='--', color='gray')
axs[1].set_xlim(-15, 15)
axs[1].set_xlabel('Lag (days)')
#plt.savefig('correlations2.pdf', bbox_inches='tight')
#plt.close('all')