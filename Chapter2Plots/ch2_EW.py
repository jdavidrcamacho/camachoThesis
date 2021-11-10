import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.linewidth'] = 2
plt.close('all')

n = np.linspace(-10,10, 1000)

from scipy.stats import norm

m, std = 0, 1.25
func = 1 - norm(m, std).pdf(n)

funcMin, funcMax = np.min(func), np.max(func)
funcHM = funcMax - (funcMax - funcMin)/2 
fwhm = 2*np.sqrt(2*np.log(2))*std

matplotlib.rcParams['figure.figsize'] = [4, 2.5]
from matplotlib.patches import Rectangle

fig, axs = plt.subplots(nrows=1, ncols=1)
axs.set_ylabel('Flux')
axs.set_xlabel('$\lambda$')
axs.plot(n,  func, 'k', linewidth=2)
axs.set_ylim(0.625, 1.025)
axs.set_xlim(-5, 5)
axs.axes.xaxis.set_ticks([])
axs.axes.yaxis.set_ticks([])
axs.add_patch(Rectangle((-1.334, 0), 2.667, 1, color ='gray') )
  
axs.hlines(y=1, xmin=-5, xmax=5, 
            colors='gray', ls='--', lw=1)
axs.hlines(y=1, xmin=-1.34, xmax=2.668-1.32, 
            colors='red', ls='-', lw=2)
# axs.vlines(x=1, ymin=0.25, ymax=np.min(func), 
#            colors='green', ls=':', lw=2)
# axs.vlines(x=1, ymin=np.min(func) + 0.01, ymax=1 - 0.01, 
#            colors='blue', ls='-', lw=2)
# axs.axes.get_yaxis().set_visible(False)

bbox_args = dict(boxstyle='round', facecolor='whitesmoke', 
                  alpha=1.00, edgecolor='red')
arrow_args = dict(arrowstyle="->", edgecolor='red')
axs.annotate('Equivalent width', xy=(0.5, 0.83), xycoords='figure fraction',
              xytext=(75, -50), textcoords='offset points',
              ha="right", va="top",
              bbox=bbox_args,
              arrowprops=arrow_args)

# bbox_args = dict(boxstyle='round', facecolor='white', 
#                  alpha=1.00, edgecolor='red')
# arrow_args = dict(arrowstyle="->", edgecolor='red')
# axs.annotate('Full width at half maximum', xy=(0.55, 0.57), xycoords='figure fraction',
#              xytext=(120, -25), textcoords='offset points',
#              ha="right", va="top",
#              bbox=bbox_args,
#              arrowprops=arrow_args)

# bbox_args = dict(boxstyle='round', facecolor='white', 
#                  alpha=1.00, edgecolor='blue')
# arrow_args = dict(arrowstyle="->", edgecolor='blue')
# axs.annotate('Bissector', xy=(0.51, 0.7), xycoords='figure fraction',
#              xytext=(-50, -50), textcoords='offset points',
#              ha="right", va="top",
#              bbox=bbox_args,
#              arrowprops=arrow_args)


plt.savefig('ewPlot.pdf', bbox_inches='tight')
