import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')


x = [0, 1, 2, 3, 4, 5, 6]
y = [1, 1, 0.8, 1.2, 0.8, 1, 1]


plt.rc_context({'xtick.color':'white', 
                'ytick.color':'white', 'figure.facecolor':'white'})

plt.rcParams['figure.figsize'] = [10/1.5, 3]
fig, axs = plt.subplots(nrows=1, ncols=1)
axs.plot(x, y, 'o:b', linewidth=2)
axs.set_xlim(0.1, 5.9)
axs.set_ylim(0.5, 1.5)
# axs.axes.get_yaxis().set_visible(False)
# axs.axes.get_xaxis().set_visible(False)

# axs.set_xticklabels(['1', '2', '3', '4', '5'])
# axs.set_yticklabels(['', '', '', '$y_1$', '', ''])
axs.tick_params(colors='white', which='major')  # 'both' refers to minor and major axes
props = dict(boxstyle='round', facecolor='white', alpha=0.00, edgecolor='white')

textstr = '$y_1$'
axs.text(-0.03, 0.51, textstr, transform=axs.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

props = dict(boxstyle='round', facecolor='white', alpha=1.00, edgecolor='black')

textstr1 = '1'
axs.text(0.15, -0.03, textstr1, transform=axs.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

textstr2 = '2'
axs.text(0.32, -0.03, textstr2, transform=axs.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

textstr3 = '3'
axs.text(0.5, -0.03, textstr3, transform=axs.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

textstr4 = '4'
axs.text(0.68, -0.03, textstr4, transform=axs.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

textstr5 = '5'
axs.text(0.85, -0.025, textstr5, transform=axs.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
plt.savefig('fwhmDiagrams.pdf', bbox_inches='tight')