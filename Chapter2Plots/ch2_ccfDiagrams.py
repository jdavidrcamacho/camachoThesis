import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
import matplotlib as mpl
plt.close('all')

n = np.linspace(-10,10, 1000)
nplus1 = n+1
lwidth = 1.5
lwidthmiddle = 1

from scipy.stats import norm
m, std = 0, 2
gauss = norm(m, std).pdf(n)
func = 1 - gauss

funcMin, funcMax = np.min(func), np.max(func)
funcHM = funcMax - (funcMax - funcMin)/2 
fwhm = 2*np.sqrt(2*np.log(2))*std

plt.rcParams['figure.figsize'] = [2*5, 2*2]
fig, axs = plt.subplots(nrows=2, ncols=5)

### "Sun"
sun0 = mpl.patches.Circle((0.5, 0.5), 0.4, color='gold')
axs[0,0].add_patch(sun0)
axs[0,0].axes.get_yaxis().set_visible(False)
axs[0,0].axes.get_xaxis().set_visible(False)
sun1 = mpl.patches.Circle((0.5, 0.5), 0.4, color='gold')
axs[0,1].add_patch(sun1)
axs[0,1].axes.get_yaxis().set_visible(False)
axs[0,1].axes.get_xaxis().set_visible(False)
sun2 = mpl.patches.Circle((0.5, 0.5), 0.4, color='gold')
axs[0,2].add_patch(sun2)
axs[0,2].axes.get_yaxis().set_visible(False)
axs[0,2].axes.get_xaxis().set_visible(False)
sun3 = mpl.patches.Circle((0.5, 0.5), 0.4, color='gold')
axs[0,3].add_patch(sun3)
axs[0,3].axes.get_yaxis().set_visible(False)
axs[0,3].axes.get_xaxis().set_visible(False)
sun4 = mpl.patches.Circle((0.5, 0.5), 0.4, color='gold')
axs[0,4].add_patch(sun4)
axs[0,4].axes.get_yaxis().set_visible(False)
axs[0,4].axes.get_xaxis().set_visible(False)

### "spot"
spot1 = mpl.patches.Circle((0.215, 0.7), 0.05, color='black')
axs[0,1].add_patch(spot1)
axs[0,1].axes.get_yaxis().set_visible(False)
axs[0,1].axes.get_xaxis().set_visible(False)
spot2 = mpl.patches.Circle((0.5, 0.7), 0.05, color='black')
axs[0,2].add_patch(spot2)
axs[0,2].axes.get_yaxis().set_visible(False)
axs[0,2].axes.get_xaxis().set_visible(False)
spot3 = mpl.patches.Circle((0.785, 0.7), 0.05, color='black')
axs[0,3].add_patch(spot3)
axs[0,3].axes.get_yaxis().set_visible(False)
axs[0,3].axes.get_xaxis().set_visible(False)

### initial CCFs 
axs[1,0].plot(n+1,  func, '-k', linewidth=lwidth)
axs[1,0].set_ylim(0.75, 1.025)
axs[1,0].axes.get_yaxis().set_visible(False)
axs[1,0].axes.get_xaxis().set_visible(False)
axs[1,0].hlines(y=funcHM, xmin=(1 - fwhm/2), xmax=(1 + fwhm/2), 
           colors='red', ls='-', lw=1)
axs[1,0].vlines(x=1, ymin=np.min(func), ymax=1, 
           colors='blue', ls='-', lw=1)
axs[1,1].plot(n+1,  func, ':k', linewidth=lwidthmiddle)
axs[1,1].set_ylim(0.75, 1.025)
axs[1,1].axes.get_yaxis().set_visible(False)
axs[1,1].axes.get_xaxis().set_visible(False)
axs[1,1].hlines(y=funcHM, xmin=(1 - fwhm/2), xmax=(1 + fwhm/2), 
           colors='red', ls=':', lw=lwidthmiddle)
axs[1,1].vlines(x=1, ymin=np.min(func), ymax=1, 
           colors='blue', ls=':', lw=lwidthmiddle)
axs[1,2].plot(n+1,  func, ':k', linewidth=lwidthmiddle)
axs[1,2].set_ylim(0.75, 1.025)
axs[1,2].axes.get_yaxis().set_visible(False)
axs[1,2].axes.get_xaxis().set_visible(False)
axs[1,2].hlines(y=funcHM, xmin=(1 - fwhm/2), xmax=(1 + fwhm/2), 
           colors='red', ls=':', lw=lwidthmiddle)
axs[1,2].vlines(x=1, ymin=np.min(func), ymax=1, 
           colors='blue', ls=':', lw=lwidthmiddle)
axs[1,3].plot(n+1,  func, ':k', linewidth=lwidthmiddle)
axs[1,3].set_ylim(0.75, 1.025)
axs[1,3].axes.get_yaxis().set_visible(False)
axs[1,3].axes.get_xaxis().set_visible(False)
axs[1,3].hlines(y=funcHM, xmin=(1 - fwhm/2), xmax=(1 + fwhm/2), 
           colors='red', ls=':', lw=lwidthmiddle)
axs[1,3].vlines(x=1, ymin=np.min(func), ymax=1, 
           colors='blue', ls=':', lw=lwidthmiddle)
axs[1,4].plot(n+1,  func, '-k', linewidth=lwidth)
axs[1,4].set_ylim(0.75, 1.025)
axs[1,4].axes.get_yaxis().set_visible(False)
axs[1,4].axes.get_xaxis().set_visible(False)
axs[1,4].hlines(y=funcHM, xmin=(1 - fwhm/2), xmax=(1 + fwhm/2), 
           colors='red', ls='-', lw=1)
axs[1,4].vlines(x=1, ymin=np.min(func), ymax=1, 
            colors='blue', ls='-', lw=1)

### altered CCFs (2nd)
g1 = norm(m, std).pdf(n)
# for ii in range(300, 451):
#     g1[ii] = 0.5*g1[ii]
    
firsst = np.linspace(g1[251], g1[330], 149)
g11 = np.concatenate((g1[0:251], firsst, g1[400:]))
f1 = 1 - g11

funcMin1, funcMax1 = np.min(f1), np.max(f1)
funcHM1 = funcMax1 - (funcMax1 - funcMin1)/2 
fwhm1 = 2*np.sqrt(2*np.log(2))*std


axs[1,1].plot(nplus1, f1, '-k', linewidth=lwidth)
axs[1,1].set_ylim(0.75, 1.025)
axs[1,1].axes.get_yaxis().set_visible(False)
axs[1,1].axes.get_xaxis().set_visible(False)
axs[1,1].hlines(y=funcHM1, xmin=(1 - fwhm1/2), xmax=(1 + fwhm1/2), 
            colors='red', ls='-', lw=1)

x = [1, 1, 2.1, 0.1]
y = [np.min(f1), 0.87, 0.95, 1]
axs[1,1].plot(x, y, '-b')
# axs[1,1].vlines(x=1, ymin=np.min(f1), ymax=1, 
#             colors='blue', ls='-', lw=1)


### altered CCFs (3rd)
gauss2 = norm(m, std).pdf(n)
nn = 0
while nn<1000:
    if nn<400 or nn>600:
        gauss2[nn] = gauss[nn]
    else:
        gauss2[nn] = 0.8*gauss[nn]
    nn = nn+1
    
func2 = 1 - gauss2
funcMin2, funcMax2 = np.min(func2), np.max(func2)
funcHM2 = funcMax2 - (funcMax2 - funcMin2)/2 
fwhm2 = 2*np.sqrt(2*np.log(2))*std

first = np.linspace(gauss2[301], gauss2[400], 99)
second = np.linspace(gauss2[599], gauss2[700], 100)
gauss22 = np.concatenate((gauss2[0:301], first, gauss2[400:600],
                          second, gauss2[700:] ))

func2 = 1 - gauss22
axs[1,2].plot(nplus1, func2, '-k', linewidth=lwidth)
axs[1,2].set_ylim(0.75, 1.025)
axs[1,2].axes.get_yaxis().set_visible(False)
axs[1,2].axes.get_xaxis().set_visible(False)
axs[1,2].hlines(y=funcHM2, xmin=(1 - fwhm2/2), xmax=(1 + fwhm2/2), 
            colors='red', ls='-', lw=1)
axs[1,2].vlines(x=1, ymin=np.min(func2), ymax=1, 
            colors='blue', ls='-', lw=1)

### altered CCFs (4th)
g22= np.flip(g11)
f2 = 1 - g22
funcMin2, funcMax2 = np.min(f2), np.max(f2)
funcHM2 = funcMax - (funcMax2 - funcMin2)/2 
fwhm2 = 2*np.sqrt(2*np.log(2))*std


axs[1,3].plot(nplus1, f2, '-k', linewidth=lwidth)
axs[1,3].set_ylim(0.75, 1.025)
axs[1,3].axes.get_yaxis().set_visible(False)
axs[1,3].axes.get_xaxis().set_visible(False)
axs[1,3].hlines(y=funcHM2, xmin=(1 - fwhm2/2), xmax=(1 + fwhm2/2), 
            colors='red', ls='-', lw=1)
axs[1,3].vlines(x=1, ymin=np.min(f2), ymax=1, 
            colors='blue', ls=':', lw=1)

x = [1, 1, -0.1, 1.9]
y = [np.min(f1), 0.87, 0.95, 1]
axs[1,3].plot(x, y, '-b')

plt.savefig('ccfDiagrams.pdf', bbox_inches='tight')