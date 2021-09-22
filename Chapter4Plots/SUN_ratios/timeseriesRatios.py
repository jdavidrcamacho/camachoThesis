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
                                                          skiprows = 1, unpack = True, 
                                                          usecols = (0,1,2,3,4,7,8,9,10))
val1, val1err = rv, rverr
val2, val2err = bis, biserr
val3, val3err = fw, fwerr
val4, val4err = rhk, rhkerr

val1Ratio = np.divide(val1, val1err).mean()
val2Ratio = np.divide(val2, val2err).mean()
val3Ratio = np.divide(val3, val3err).mean()
val4Ratio = np.divide(val4, val4err).mean()

print('RV:', val1Ratio)
print('BIS:', val2Ratio)
print('FWHM:', val3Ratio)
print('log Rhk:', val4Ratio)