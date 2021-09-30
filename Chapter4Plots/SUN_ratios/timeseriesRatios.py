import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')

time,rv,rverr,rhk,rhkerr,bis,biserr,fw,fwerr = np.loadtxt("sunBinned_Dumusque.txt", 
                                                          skiprows = 350, max_rows=50, unpack = True, 
                                                          usecols = (0,1,2,3,4,7,8,9,10))
val1, val1err = rv, rverr
val2, val2err = bis, biserr
val3, val3err = fw, fwerr
val4, val4err = rhk, rhkerr

val1Ratio = val1.ptp() / val1err.mean()
val2Ratio = val2.ptp() / val2err.mean()
val3Ratio = val3.ptp() / val3err.mean()
val4Ratio = val4.ptp() / val4err.mean()

# relUnc1 = val1err.std()/val1.ptp()
# relUnc2 = val2err.std()/val2.ptp()
# relUnc3 = val3err.std()/val3.ptp()
# relUnc4 = val4err.std()/val4.ptp()
# print(relUnc1,relUnc2,relUnc3,relUnc4)
# print()

print('RV:', np.mean(val1Ratio))
print('BIS:', np.mean(val2Ratio))
# print(val2Ratio/val1Ratio)
print('FWHM:', np.mean(val3Ratio))
# print(val3Ratio/val1Ratio)
print('log Rhk:', np.mean(val4Ratio))
# print(val4Ratio/val1Ratio)
print()

# val1Ratio = val1.ptp() / (np.std(val1err)/np.sqrt(2))
# val2Ratio = val2.ptp() / (np.std(val2err)/np.sqrt(2))
# val3Ratio = val3.ptp() / (np.std(val3err)/np.sqrt(2))
# val4Ratio = val4.ptp() / (np.std(val4err)/np.sqrt(2))

# print('RV:', np.mean(val1Ratio))
# print('BIS:', np.mean(val2Ratio))
# print('FWHM:', np.mean(val3Ratio))
# print('log Rhk:', np.mean(val4Ratio))