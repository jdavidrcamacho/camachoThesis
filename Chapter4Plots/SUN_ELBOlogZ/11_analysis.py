import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')


points, logz, elbo, diff = np.loadtxt("elboLogZ_results.txt", skiprows = 1, 
                                      unpack = True, usecols = (0, 1, 2, 3))

first50 = points[0:50]
first50logz = logz[0:50]
first50elbo = elbo[0:50]

plt.rcParams['figure.figsize'] = [15, 5]
plt.figure()
plt.plot(first50, first50elbo, '.--b', label = 'ELBO')
plt.plot(first50, first50logz, '.-r', label = 'log Z')
plt.ylabel('Value')
plt.xlabel('Points')
plt.legend(facecolor='white', framealpha=1, edgecolor='black')


plt.rcParams['figure.figsize'] = [7, 3]

#fig, axs = plt.subplots(2,1, sharex=True, gridspec_kw={'width_ratios': [1],
#                                                       'height_ratios': [2, 1]})
#axs[0].set_xscale('log')
#axs[0].plot(points[[0,1,2,3,4,5,6,7,8,9, 19, 29, 39, 49, 50, 51]], 
#            elbo[[0,1,2,3,4,5,6,7,8,9, 19, 29, 39, 49, 50, 51]],
#            '.-r', label = 'ELBO', linewidth=2)
#axs[0].plot(points[[0,1,2,3,4,5,6,7,8,9, 19, 29, 39, 49, 50, 51]], 
#            logz[[0,1,2,3,4,5,6,7,8,9, 19, 29, 39, 49, 50, 51]],
#            '.:b', label = '$\log \mathcal{Z}$', linewidth=2)
#axs[0].set_ylabel('Value')
#axs[0].legend(facecolor='white', framealpha=1, edgecolor='black')
#axs[0].tick_params(axis='both', which='both', labelbottom=False)
#axs[1].plot(points[[0,1,2,3,4,5,6,7,8,9, 19, 29, 39, 49, 50, 51]], 
#            diff[[0,1,2,3,4,5,6,7,8,9, 19, 29, 39, 49, 50, 51]],
#            '.-k', linewidth=2)
#axs[1].axhline(y = meanDiff, linestyle='--', color='k', linewidth=1)
#axs[1].set_ylabel('$\log \mathcal{Z}$  - ELBO')
#axs[1].set_xlabel('Points')
#axs[1].tick_params(axis='both', which='both', labelbottom=True)
#plt.savefig('elboAnalysis.pdf', bbox_inches='tight')
# plt.close('all')

meanDiff = np.mean(diff)
fig, axs = plt.subplots(2,1, sharex=True, gridspec_kw={'width_ratios': [1],
                                                       'height_ratios': [2, 1]})
axs[0].set_xscale('log')
axs[0].plot(points, elbo, '.-r', label = 'ELBO', linewidth=2)
axs[0].plot(points, logz, '.:b', label = '$\log \mathcal{Z}$', linewidth=2)
axs[0].set_ylabel('Value')
axs[0].legend(facecolor='white', framealpha=1, edgecolor='black')
axs[0].tick_params(axis='both', which='both', labelbottom=False)
axs[1].plot(points,  diff, '.k', linewidth=2)
axs[1].axhline(y = meanDiff, linestyle='--', color='k', linewidth=1)
axs[1].set_ylabel('$\log \mathcal{Z}$  - ELBO')
axs[1].set_xlabel('Number of data points')
axs[1].tick_params(axis='both', which='both', labelbottom=True)
plt.savefig('elboAnalysis.pdf', bbox_inches='tight')
