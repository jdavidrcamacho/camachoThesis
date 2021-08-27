import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')

n = np.linspace(-10,10, 1000)

from scipy.stats import norm

m, std = 0, 1.25
func = 1 - norm(m, std).pdf(n)

funcMin, funcMax = np.min(func), np.max(func)
funcHM = funcMax - (funcMax - funcMin)/2 
fwhm = 2*np.sqrt(2*np.log(2))*std

fig, axs = plt.subplots(nrows=1, ncols=1)
axs.set_ylabel('CCF')
axs.set_xlabel('Velocity (m/s)')
axs.plot(n+1,  func, 'k', linewidth=2)
axs.set_ylim(0.625, 1.025)

axs.hlines(y=funcHM, xmin=(1 - fwhm/2), xmax=(1 + fwhm/2), 
           colors='red', ls='--', lw=2)
axs.vlines(x=1, ymin=0.25, ymax=np.min(func), 
           colors='green', ls=':', lw=2)
axs.vlines(x=1, ymin=np.min(func) + 0.01, ymax=1 - 0.01, 
           colors='blue', ls='-', lw=2)
axs.axes.get_yaxis().set_visible(False)

bbox_args = dict(boxstyle='round', facecolor='white', 
                 alpha=1.00, edgecolor='green')
arrow_args = dict(arrowstyle="->", edgecolor='green')
axs.annotate('Radial velocity', xy=(0.52, 0.125), xycoords='figure fraction',
             xytext=(75, 25), textcoords='offset points',
             ha="right", va="top",
             bbox=bbox_args,
             arrowprops=arrow_args)

bbox_args = dict(boxstyle='round', facecolor='white', 
                 alpha=1.00, edgecolor='red')
arrow_args = dict(arrowstyle="->", edgecolor='red')
axs.annotate('Full width at half maximum', xy=(0.55, 0.57), xycoords='figure fraction',
             xytext=(120, -25), textcoords='offset points',
             ha="right", va="top",
             bbox=bbox_args,
             arrowprops=arrow_args)

bbox_args = dict(boxstyle='round', facecolor='white', 
                 alpha=1.00, edgecolor='blue')
arrow_args = dict(arrowstyle="->", edgecolor='blue')
axs.annotate('Bissector', xy=(0.51, 0.7), xycoords='figure fraction',
             xytext=(-50, -50), textcoords='offset points',
             ha="right", va="top",
             bbox=bbox_args,
             arrowprops=arrow_args)


plt.savefig('ccfPlot.pdf', bbox_inches='tight')