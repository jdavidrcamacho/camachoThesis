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

from scipy.stats import norm
x = np.linspace(-5, 5, 1000)

plt.rcParams['figure.figsize'] = [7, 3]
plt.figure()
y = norm(0, 1).pdf(x)
plt.fill_between(x, y, color='b', alpha = 0.50)

y = norm(2, 0.5).pdf(x)
plt.fill_between(x, y, hatch='oo', color='r', alpha = 0.50)

y = norm(-1, 2).pdf(x)
plt.fill_between(x, y, hatch='++', color='g', alpha = 0.50)


plt.xlabel('x')
plt.ylabel('probability')
plt.savefig('gaussDistribution.png', bbox_inches='tight')
plt.savefig('gaussDistribution.pdf', bbox_inches='tight')
