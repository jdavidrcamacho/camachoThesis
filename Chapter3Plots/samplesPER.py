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

from tedi import kernels, process, means

time = np.array([2, 5]) 
val1 = [-1, 1]
val1err = 0.1*np.ones_like(val1)

mean = means.Constant(0)

tstar= np.linspace(0, 2, 1000)

plt.rcParams['figure.figsize'] = [7, 3]
plt.figure()

eta1, eta3, eta4 = 1, 1, 0.1
kernel1 = kernels.Periodic(eta1, eta3, eta4)
tedibear1 = process.GP(kernel1, mean, time, val1, val1err)
a1 = tedibear1.sample(kernel1, tstar)
plt.plot(tstar, a1, color='blue', linestyle='solid', linewidth=2, 
          label='$\eta_4 = 0.1$')

eta1, eta3, eta4 = 1, 1, 1
kernel2 = kernels.Periodic(eta1, eta3, eta4)
tedibear2 = process.GP(kernel2, mean, time, val1, val1err)
a2 = tedibear2.sample(kernel2, tstar)
plt.plot(tstar, a2, color='red', linestyle='dashed', linewidth=2, 
          label='$\eta_4 = 1$')

eta1, eta3, eta4 = 1, 1, 10
kernel3 = kernels.Periodic(eta1, eta3, eta4)
tedibear3 = process.GP(kernel3, mean, time, val1, val1err)
a3 = tedibear3.sample(kernel3, tstar)
plt.plot(tstar, a3, color = 'green', linestyle='dotted', linewidth=2, 
          label='$\eta_4 = 10$')
plt.ylabel('Output')
plt.xlabel('Input')
plt.legend(facecolor='whitesmoke', framealpha=1, edgecolor='black')
plt.savefig('samplesPerKernel.pdf', bbox_inches='tight')
