import numpy as np
import matplotlib
#matplotlib.rcParams['figure.figsize'] = [15, 2*5]
#matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})

import matplotlib.pyplot as plt
plt.close('all')


###### Data .rdb file #####
time,rv,rverr,rhk,rhkerr,bis,biserr,fw,fwerr = np.loadtxt("sunBinned_Dumusque.txt", 
                                                           skiprows = 1, 
                                                           unpack = True, 
                                                           usecols = (0, 1, 2, 3, 4,
                                                                      7, 8, 9, 10))
time = time


# Trend removal ################################################################
rvFit = np.poly1d(np.polyfit(time, rv, 1))
rv = np.array(rv)-rvFit(time)

bisFit = np.poly1d(np.polyfit(time, bis, 1))
bis = np.array(bis)-bisFit(time)

fwFit = np.poly1d(np.polyfit(time, fw, 1))
fw = np.array(fw)-fwFit(time)

rhkFit = np.poly1d(np.polyfit(time, rhk, 1))
rhk = np.array(rhk)-rhkFit(time)


fig, axs = plt.subplots(nrows=4, ncols=2)
fig.set_size_inches(w=2*4.15, h=4.5+1)

axs[0,0].errorbar(time, rv, rverr, fmt= '.b', markersize =2, elinewidth=1)
axs[0,0].tick_params(axis='both', which='both', labelbottom=False)
axs[0,0].set_ylabel('RVs (m/s)')

axs[1,0].errorbar(time, bis, biserr, fmt= '.b', markersize =2, elinewidth=1)
axs[1,0].set_ylabel('BIS (m/s)')
axs[1,0].tick_params(axis='both', which='both', labelbottom=False)

axs[2,0].errorbar(time, fw, fwerr, fmt= '.b', markersize =2, elinewidth=1)
axs[2,0].set_ylabel('FWHM (m/s)')
axs[2,0].tick_params(axis='both', which='both', labelbottom=False)

axs[3,0].errorbar(time, rhk, rhkerr, fmt= '.b', markersize =2, elinewidth=1)
axs[3,0].set_ylabel('$\log R^{\'}_{hk}$')
axs[3,0].set_xlabel('Time (JDB-2400000.0)')
axs[3,0].tick_params(axis='both', which='both', labelbottom=True)


from astropy.timeseries import LombScargle
linesize = 1


per = 196.560


f1, p1 = LombScargle(time, rv, rverr).autopower()
axs[0,1].semilogx(1/f1, p1, color='blue', linewidth=linesize)
bestf = f1[np.argmax(p1)]
bestp = 1/bestf
axs[0,1].axvline(x=27, ymin=p1.min(), ymax=100*p1.max(), color='red', alpha=1,
               linewidth=linesize)
axs[0,1].axvline(x=per, ymin=p1.min(), ymax=100*p1.max(), color='orange', alpha=1,
               linewidth=linesize)
axs[0,1].axvline(x=365.256/2.0, ymin=p1.min(), ymax=100*p1.max(), color='cyan', alpha=1,
               linewidth=linesize)
axs[0,1].tick_params(axis='both', which='both', labelbottom=False)

#false alarm
falseAlarms1 = LombScargle(time, rv, rverr).false_alarm_level([0.1,0.01,0.001])
one = falseAlarms1[1] * np.ones_like(f1)
axs[0,1].plot(1/f1, one, color='red', linestyle='dashed', alpha=1,linewidth=linesize)
axs[0,1].set_xlim(1,time.ptp())

f2, p2 = LombScargle(time, bis, biserr).autopower()
axs[1,1].semilogx(1/f2, p2, color='blue', linewidth=linesize)
axs[1,1].yaxis.set_label_coords(-0.1, -.2)
axs[1,1].set_ylabel('Normalized power')
bestf = f2[np.argmax(p2)]
bestp = 1/bestf
axs[1,1].axvline(x=27, ymin=p2.min(), ymax=100*p2.max(), color='red', alpha=1,
               linewidth=linesize)
axs[1,1].axvline(x=per, ymin=p1.min(), ymax=100*p1.max(), color='orange', alpha=1,
               linewidth=linesize)
axs[1,1].tick_params(axis='both', which='both', labelbottom=False)
#false alarm
falseAlarms2 = LombScargle(time, bis, biserr).false_alarm_level([0.1,0.01,0.001])
one = falseAlarms2[1] * np.ones_like(f2)
axs[1,1].plot(1/f2, one, color='r', linestyle='dashed', alpha=1, linewidth=linesize)
axs[1,1].set_xlim(1,time.ptp())

f3, p3 = LombScargle(time, fw, fwerr).autopower()
falseAlarms3 = LombScargle(time, fw, fwerr).false_alarm_level([0.1,0.01,0.001])
axs[2,1].semilogx(1/f3, p3, color='blue', linewidth=linesize)
bestf = f3[np.argmax(p3)]
bestp = 1/bestf
axs[2,1].axvline(x=27, ymin=p3.min(), ymax=100*p3.max(), color='r', alpha=1,
               linewidth=linesize)
axs[2,1].axvline(x=per, ymin=p1.min(), ymax=100*p1.max(), color='orange', alpha=1,
               linewidth=linesize)
axs[2,1].tick_params(axis='both', which='both', labelbottom=False)
falseAlarms3 = LombScargle(time, fw, fwerr).false_alarm_level([0.1,0.01,0.001])
one = falseAlarms3[1] * np.ones_like(f3)
axs[2,1].plot(1/f3, one, color='red', linestyle='dashed', alpha=1, linewidth=linesize)
axs[2,1].set_xlim(1,time.ptp())

f4, p4 = LombScargle(time, rhk, rhkerr).autopower()
falseAlarms3 = LombScargle(time, rhk, rhkerr).false_alarm_level([0.1,0.01,0.001])
axs[3,1].semilogx(1/f4, p4, color='blue', linewidth=linesize)
bestf = f4[np.argmax(p4)]
bestp = 1/bestf
axs[3,1].axvline(x=27, ymin=p4.min(), ymax=100*p4.max(), color='red', alpha=1,
               linewidth=linesize)
axs[3,1].axvline(x=per, ymin=p1.min(), ymax=100*p1.max(), color='orange', alpha=1,
               linewidth=linesize)
axs[3,1].tick_params(axis='both', which='both', labelbottom=True)
axs[3,1].set_xlabel('Period (days)')
#false alarm
falseAlarms4 = LombScargle(time, rhk, rhkerr).false_alarm_level([0.1,0.01,0.001])
one = falseAlarms4[1] * np.ones_like(f4)
axs[3,1].plot(1/f4, one, color='red', linestyle='dashed', alpha=1, linewidth=linesize)
axs[3,1].set_xlim(1,time.ptp())

plt.tight_layout(h_pad=0.7, w_pad=0.7)
plt.savefig('01_periodograms_b.pdf', bbox_inches='tight')
plt.close('all')



# ###### Data .rdb file #####
# time,rv,rverr,bis,biserr,fw,fwerr = np.loadtxt("sunBinned_Dumusque.txt", 
#                                                            skiprows = 1, 
#                                                            unpack = True, 
#                                                            usecols = (0,1,2,
#                                                                       7,8,9,10))
# time = time


# # Trend removal ################################################################
# rvFit = np.poly1d(np.polyfit(time, rv, 1))
# rv = np.array(rv)-rvFit(time)

# bisFit = np.poly1d(np.polyfit(time, bis, 1))
# bis = np.array(bis)-bisFit(time)

# fwFit = np.poly1d(np.polyfit(time, fw, 1))
# fw = np.array(fw)-fwFit(time)


# fig, axs = plt.subplots(nrows=3, ncols=2)
# fig.set_size_inches(w=2*4.15, h=3.5)

# axs[0,0].errorbar(time, rv, rverr, fmt= '.k', markersize =2, elinewidth=1)
# axs[0,0].tick_params(axis='both', which='both', labelbottom=False)
# axs[0,0].set_ylabel('RVs (m/s)')

# axs[1,0].errorbar(time, bis, biserr, fmt= '.k', markersize =2, elinewidth=1)
# axs[1,0].set_ylabel('BIS (m/s)')
# axs[1,0].tick_params(axis='both', which='both', labelbottom=False)

# axs[2,0].errorbar(time, fw, fwerr, fmt= '.k', markersize =2, elinewidth=1)
# axs[2,0].set_ylabel('FWHM (m/s)')
# axs[2,0].tick_params(axis='both', which='both', labelbottom=False)
# axs[2,0].set_xlabel('Time (JDB-2400000.0)')
# axs[2,0].tick_params(axis='both', which='both', labelbottom=True)


# linesize = 1

# f1, p1 = LombScargle(time, rv, rverr).autopower()
# axs[0,1].semilogx(1/f1, p1, color='black', linewidth=linesize)
# bestf = f1[np.argmax(p1)]
# bestp = 1/bestf
# axs[0,1].axvline(x=27, ymin=p1.min(), ymax=100*p1.max(), color='r', alpha=0.75,
#                linewidth=linesize)
# axs[0,1].tick_params(axis='both', which='both', labelbottom=False)
# #false alarm
# falseAlarms1 = LombScargle(time, rv, rverr).false_alarm_level([0.1,0.01,0.001])
# one = falseAlarms1[1] * np.ones_like(f1)
# axs[0,1].plot(1/f1, one, color='r', linestyle='dashed', alpha=0.75,linewidth=linesize)
# axs[0,1].set_xlim(1,time.ptp())

# f2, p2 = LombScargle(time, bis, biserr).autopower()
# axs[1,1].semilogx(1/f2, p2, color='black', linewidth=linesize)
# axs[1,1].set_ylabel('Normalized power')
# bestf = f2[np.argmax(p2)]
# bestp = 1/bestf
# axs[1,1].axvline(x=27, ymin=p2.min(), ymax=100*p2.max(), color='r', alpha=0.75,
#                linewidth=linesize)
# axs[1,1].tick_params(axis='both', which='both', labelbottom=False)
# #false alarm
# falseAlarms2 = LombScargle(time, bis, biserr).false_alarm_level([0.1,0.01,0.001])
# one = falseAlarms2[1] * np.ones_like(f2)
# axs[1,1].plot(1/f2, one, color='r', linestyle='dashed', alpha=0.75, linewidth=linesize)
# axs[1,1].set_xlim(1,time.ptp())

# f3, p3 = LombScargle(time, fw, fwerr).autopower()
# falseAlarms3 = LombScargle(time, fw, fwerr).false_alarm_level([0.1,0.01,0.001])
# axs[2,1].semilogx(1/f3, p3, color='black', linewidth=linesize)
# bestf = f3[np.argmax(p3)]
# bestp = 1/bestf
# axs[2,1].axvline(x=27, ymin=p3.min(), ymax=100*p3.max(), color='r', alpha=0.75,
#                linewidth=linesize)
# axs[2,1].tick_params(axis='both', which='both', labelbottom=True)
# falseAlarms3 = LombScargle(time, fw, fwerr).false_alarm_level([0.1,0.01,0.001])
# one = falseAlarms3[1] * np.ones_like(f3)
# axs[2,1].plot(1/f3, one, color='r', linestyle='dashed', alpha=0.75, linewidth=linesize)
# axs[2,1].set_xlim(1,time.ptp())
# axs[2,1].tick_params(axis='both', which='both', labelbottom=True)
# axs[2,1].set_xlabel('Period (days)')

# plt.tight_layout(h_pad=0.7, w_pad=0.7)
# plt.savefig('01_periodograms_RVsBISandFWHM.pdf', bbox_inches='tight')
# plt.close('all')

# print(time.ptp())
