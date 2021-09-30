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
                                                          skiprows = 1, unpack = True, 
                                                          usecols = (0,1,2,3,4,7,8,9,10))
val1, val1err = rv, np.mean(rverr)
val2, val2err = bis, np.mean(biserr)
val3, val3err = fw, np.mean(fwerr)
val4, val4err = rhk, np.mean(rhkerr)


val11 = val1 
val22 = val2 
val33 = val3 
val44 = val4 

val22 = val22 - np.polyval(np.polyfit(time, val22, 1), time)
val33 = val33 - np.polyval(np.polyfit(time, val33, 1), time)
val44 = val44 - np.polyval(np.polyfit(time, val44, 1), time)

plt.rcParams['figure.figsize'] = [3, 3]
fig, axs = plt.subplots(1, 1)

from correlation_functions import DCF_EK

axs.xaxis.set_minor_locator(AutoMinorLocator(5))
axs.yaxis.set_minor_locator(AutoMinorLocator(5))
axs.grid(which='major', alpha=0.5)
axs.grid(which='minor', alpha=0.2)

EKbins = np.linspace(-15, 15, 31)
C_EK, C_EK_err, bins = DCF_EK(time, val11, val22, val2err, val2err, bins=EKbins)
t_EK = 0.5 * (bins[1:] + bins[:-1])
m = ~np.isnan(C_EK)
axs.plot(t_EK[m], C_EK[m],  '-.b', label='RV and BIS')

EKbins = np.linspace(-15, 15, 31)
C_EK, C_EK_err, bins = DCF_EK(time, val11, val33, val3err, val3err, bins=EKbins)
t_EK = 0.5 * (bins[1:] + bins[:-1])
m = ~np.isnan(C_EK)
axs.plot(t_EK[m], C_EK[m], '--r', label='RV and FWHM')

EKbins = np.linspace(-15, 15, 31)
C_EK, C_EK_err, bins = DCF_EK(time, val11, val44, val4err, val4err, bins=EKbins)
t_EK = 0.5 * (bins[1:] + bins[:-1])
m = ~np.isnan(C_EK)
axs.plot(t_EK[m], C_EK[m], ':g', label='RV and $\log R^{\'}_{hk}$')
axs.legend(loc='upper right', facecolor='white', framealpha=1, edgecolor='black')
axs.set_xlim(-15, 15)
axs.grid(True)
axs.set_ylabel('Cross-correlated signal')
axs.set_xlabel('Lag (days)')
plt.savefig('data_lags.pdf', bbox_inches='tight')
# plt.close('all')

