import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')

from tedi import kernels, process, means

time = np.array([2, 5]) 
val1 = [-1, 1]
val1err = 0.1*np.ones_like(val1)

mean = means.Constant(0)

tstar= np.linspace(0, 100, 1000)

plt.rcParams['figure.figsize'] = [10/1.5, 3]
plt.figure()

kernel1 = kernels.SquaredExponential(1, 1)
tedibear1 = process.GP(kernel1, mean, time, val1, val1err)
a1 = tedibear1.sample(kernel1, tstar)
plt.plot(tstar, a1, color='blue', linestyle='solid', linewidth=2, 
         label='$\eta_2 = 1$')

kernel2 = kernels.SquaredExponential(1, 10)
tedibear2 = process.GP(kernel2, mean, time, val1, val1err)
a2 = tedibear2.sample(kernel2, tstar)
plt.plot(tstar, a2, color='red', linestyle='dashed', linewidth=2, 
         label='$\eta_2 = 10$')

kernel3 = kernels.SquaredExponential(1, 100)
tedibear3 = process.GP(kernel3, mean, time, val1, val1err)
a3 = tedibear3.sample(kernel3, tstar)
plt.plot(tstar, a3, color = 'green', linestyle='dotted', linewidth=2, 
         label='$\eta_2 = 100$')

plt.ylabel('Output')
plt.xlabel('Input')
plt.legend(facecolor='white', framealpha=1, edgecolor='black')
plt.savefig('samplesSEKernel.pdf', bbox_inches='tight')
