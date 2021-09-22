import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
#plt.close('all')

###### Data .rdb file #####
time,rv,rverr,fw,bis = np.loadtxt("table1.dat", skiprows = 0,
                              unpack = True, usecols = (0,1,2,3,4))
val1, val1err = 1*(rv-rv.mean()), rverr
val2, val2err = bis, 2.0*val1err
val3, val3err = 1*(fw-fw.mean()), 2.35*val1err

from scipy import signal
val11 = val1 - np.mean(val1)
val22 = val2 - np.mean(val2)
val33 = val3 - np.mean(val3)

corr1 = signal.correlate(val11, val22, mode='full', method='direct')
lags1 = signal.correlation_lags(len(val11), len(val22), mode='full')
corr1 = corr1 / (val11.size * val1.std() * val2.std())
#corr1 /= np.max(corr21)

corr2 = signal.correlate(val11, val33, mode='full', method='direct')
lags2 = signal.correlation_lags(len(val11), len(val33), mode='full')
corr2 = corr2 / (val11.size * val1.std() * val3.std())
#corr2 /= np.max(corr2)

from matplotlib.ticker import AutoMinorLocator
plt.rcParams['figure.figsize'] = [7, 3]
fig, axs = plt.subplots(1, 1)
axs.xaxis.set_minor_locator(AutoMinorLocator(5))
axs.yaxis.set_minor_locator(AutoMinorLocator(5))
axs.grid(which='major', alpha=0.5)
axs.grid(which='minor', alpha=0.2)
# axs.plot(lags1, corr1, '--r', label='Scipy - RV BIS')
axs.plot(lags2, corr2, '--b', label='Scipy - RV FW')
axs.set_xlim(-15, 15)
axs.set_ylabel('Cross-correlated signal')
axs.set_xlabel('')
axs.tick_params(axis='both', which='both', labelbottom=True)
# plt.close('all')

# val1, val1err = 1000*rv, rverr
# val2, val2err = bis, 2.0*val1err
# val3, val3err = 1000*fw, 2.35*val1err
from correlation_functions import DCF_EK

EKbins = np.linspace(-15, 15, 31)

# C_EK, C_EK_err, bins = DCF_EK(time, val11, val22, 
#                               val1err, val2err, bins=EKbins)
# t_EK = 0.5 * (bins[1:] + bins[:-1])
# m = ~np.isnan(C_EK)
# axs.plot(t_EK[m], C_EK[m], '-r', label='Edelson-Krolik - RV BIS')

C_EK, C_EK_err, bins = DCF_EK(time, val1, val3, 
                              val1err, val3err, bins=EKbins)
t_EK = 0.5 * (bins[1:] + bins[:-1])
m = ~np.isnan(C_EK)
axs.plot(t_EK[m], C_EK[m], '-b', label='Edelson-Krolik - RV FW')

axs.legend(loc='upper right', facecolor='white', framealpha=1, edgecolor='black')
axs.set_xlabel('Lag (days)')
plt.savefig('correlationHD41248.pdf', bbox_inches='tight')
#plt.close('all')

# import numpy as np
# import matplotlib
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False})
# import matplotlib.pylab as plt
# plt.close('all')

# ###### Data .rdb file #####
# time,rv,rverr,fw,bis = np.loadtxt("table1.dat", skiprows = 0,
#                                   unpack = True, usecols = (0,1,2,3,4))
# val1, val1err = 1000*rv, 1000*rverr
# val2, val2err = bis, 2.0*val1err
# val3, val3err = 1000*fw, 2.35*val1err

# from scipy import signal
# val11 = val1 - np.mean(val1)
# val22 = val2 - np.mean(val2)
# val33 = val3 - np.mean(val3)

# corr1 = signal.correlate(val11, val22, mode='full', method='direct')
# lags1 = signal.correlation_lags(len(val11), len(val22), mode='full')
# corr1 = corr1 / (val11.size * val1.std() * val2.std())
# #corr2 /= np.max(corr2)
# corr2 = signal.correlate(val11, val33, mode='full', method='direct')
# lags2 = signal.correlation_lags(len(val11), len(val33), mode='full')
# corr2 = corr2 / (val11.size * val1.std() * val3.std())
# #corr2 /= np.max(corr2)

# from matplotlib.ticker import AutoMinorLocator
# plt.rcParams['figure.figsize'] = [7, 3]
# fig, axs = plt.subplots(1, 1)
# axs.xaxis.set_minor_locator(AutoMinorLocator(5))
# axs.yaxis.set_minor_locator(AutoMinorLocator(5))
# axs.grid(which='major', alpha=0.5)
# axs.grid(which='minor', alpha=0.2)
# axs.plot(lags1, corr1, '--r', label='Scipy - RV BIS')
# axs.plot(lags2, corr2, '--b', label='Scipy - RV FW')
# axs.set_xlim(-15, 15)
# axs.set_ylabel('Cross-correlated signal')
# axs.set_xlabel('')
# axs.tick_params(axis='both', which='both', labelbottom=True)
# # plt.close('all')

# # val1, val1err = 1000*rv, 1000*rverr
# # val2, val2err = bis, 2.0*val1err
# # val3, val3err = 1000*fw, 2.35*val1err
# from correlation_functions import DCF_EK

# # EKbins = np.linspace(-15, 15, 31)
# # C_EK, C_EK_err, bins = DCF_EK(time, val1, val2, val1err, val2err, bins=EKbins)
# # t_EK = 0.5 * (bins[1:] + bins[:-1])
# # m = ~np.isnan(C_EK)
# # axs.plot(t_EK[m], C_EK[m], '-r', label='Edelson-Krolik - RV BIS')


# EKbins = np.linspace(-15, 15, 31)
# C_EK, C_EK_err, bins = DCF_EK(time, val1, val3, val1err, val3err, bins=EKbins)
# t_EK = 0.5 * (bins[1:] + bins[:-1])
# m = ~np.isnan(C_EK)
# axs.plot(t_EK[m], C_EK[m], '-b', label='Edelson-Krolik - RV FW')

# axs.legend(loc='upper right', facecolor='white', framealpha=1, edgecolor='black')
# axs.set_xlabel('Lag (days)')
# plt.savefig('correlationhd41501.pdf', bbox_inches='tight')
# #plt.close('all')

# ################################################################################
# time,rv,rverr,fw = np.loadtxt("table1.dat", skiprows = 227,
#                               unpack = True, usecols = (0,1,2,3))
# val1, val1err = 1000*rv, 1000*rverr
# val3, val3err = 1000*fw, 2.35*val1err

# from scipy import signal
# val11 = val1 - np.mean(val1)
# val33 = val3 - np.mean(val3)
# corr2 = signal.correlate(val11, val33, mode='full', method='direct')
# lags2 = signal.correlation_lags(len(val11), len(val33), mode='full')
# corr2 = corr2 / (val11.size * val1.std() * val3.std())
# #corr2 /= np.max(corr2)


# from matplotlib.ticker import AutoMinorLocator
# plt.rcParams['figure.figsize'] = [7, 3]
# fig, axs = plt.subplots(1, 1)
# axs.xaxis.set_minor_locator(AutoMinorLocator(5))
# axs.yaxis.set_minor_locator(AutoMinorLocator(5))
# axs.grid(which='major', alpha=0.5)
# axs.grid(which='minor', alpha=0.2)

# axs.plot(lags2, corr2, '-b', label='Scipy method')
# axs.set_xlim(-15, 15)
# axs.set_ylabel('Cross-correlated signal')
# axs.set_xlabel('')
# axs.tick_params(axis='both', which='both', labelbottom=True)
# # plt.close('all')

# val1, val1err = rv, rverr
# val3, val3err = fw, 2.35*val1err
# from correlation_functions import DCF_EK

# EKbins = np.linspace(-15, 15, 31)
# C_EK, C_EK_err, bins = DCF_EK(time, val1, val3, val1err, val3err, bins=EKbins)
# t_EK = 0.5 * (bins[1:] + bins[:-1])
# m = ~np.isnan(C_EK)
# axs.plot(t_EK[m], C_EK[m], '-r', label='Edelson-Krolik method')

# axs.legend(loc='upper right', facecolor='white', framealpha=1, edgecolor='black')
# axs.set_xlabel('Lag (days)')
# plt.savefig('correlationhd41501.pdf', bbox_inches='tight')
# #plt.close('all')

