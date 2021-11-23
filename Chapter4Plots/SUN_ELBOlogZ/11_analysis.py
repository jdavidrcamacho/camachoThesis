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
matplotlib.rcParams['figure.figsize'] = [7, 4]


points, logz, elbo, diff = np.loadtxt("elboLogZ_results.txt", skiprows = 1, 
                                      unpack = True, usecols = (0, 1, 2, 3))

meanDiff = np.mean(diff)
print(meanDiff)

fig, axs = plt.subplots(2,1, sharex=True, gridspec_kw={'width_ratios': [1],
                                                       'height_ratios': [2, 1]})

axs[0].set_xscale('log')
axs[0].plot(points, elbo, 'o-r', label = 'ELBO', linewidth=2)
axs[0].plot(points, logz, 'o--b', label = '$\\texttt{dynesty}$', linewidth=2)
axs[0].set_ylabel('Value')
axs[1].axhline(y = meanDiff, linestyle='--', color='k', linewidth=1)

axs[0].legend(facecolor='whitesmoke', framealpha=1, edgecolor='black')
axs[0].tick_params(axis='both', which='both', labelbottom=False)
axs[1].plot(points,  diff, 'ok', linewidth=2, alpha=0.5)
axs[1].set_ylabel('$\\texttt{dynesty}$  - ELBO')
axs[1].set_xlabel('Number of data points')
axs[1].tick_params(axis='both', which='both', labelbottom=True)

axs[1].text(0.28, 0.75, 'Average difference = {0}'.format(round(meanDiff, 3)),
            bbox={'facecolor':'whitesmoke', 'alpha':1, 'boxstyle':'round'},
            verticalalignment='bottom', horizontalalignment='right',
            transform=axs[1].transAxes, color='black', fontsize=10)

plt.tight_layout(pad=0.1, h_pad=0.25, w_pad=0.1)
plt.savefig('elboAnalysis.pdf', bbox_inches='tight')
# plt.close('all')