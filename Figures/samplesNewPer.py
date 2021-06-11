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

tstar= np.linspace(0, 5, 1000)

plt.rcParams['figure.figsize'] = [10/1.5, 3]
plt.figure()

eta1, eta3, eta4 = 1, 1, 1
alpha = 1
np.random.seed(1)
kernel0 = kernels.NewPeriodic(eta1, alpha, eta3, eta4)
tedibear0 = process.GP(kernel0, mean, time, val1, val1err)
a0 = tedibear0.sample(kernel0, tstar)
plt.plot(tstar, a0, color = 'grey', linestyle='solid', linewidth=2, 
          label='$a=1$')

alpha = 10
np.random.seed(1)
kernel1 = kernels.NewPeriodic(eta1, alpha, eta3, eta4)
tedibear1 = process.GP(kernel1, mean, time, val1, val1err)
a1 = tedibear1.sample(kernel1, tstar)
plt.plot(tstar, a1, color='blue', linestyle='dashdot', linewidth=2, 
         label='$a=10$')

alpha = 101
np.random.seed(1)
kernel2 = kernels.NewPeriodic(eta1, alpha, eta3, eta4)
tedibear2 = process.GP(kernel2, mean, time, val1, val1err)
a2 = tedibear2.sample(kernel2, tstar)
plt.plot(tstar, a2, color='red', linestyle='dashed', linewidth=2, 
         label='$a=100$')

alpha = 1000
np.random.seed(1)
kernel3 = kernels.NewPeriodic(eta1, alpha, eta3, eta4)
tedibear3 = process.GP(kernel3, mean, time, val1, val1err)
a3 = tedibear3.sample(kernel3, tstar)
plt.plot(tstar, a3, color = 'green', linestyle='dashed', linewidth=2, 
          label='$a=1000$')

np.random.seed(1)
kernel4 = kernels.Periodic(eta1, eta3, eta4)
tedibear4 = process.GP(kernel4, mean, time, val1, val1err)
a4 = tedibear4.sample(kernel4, tstar)
plt.plot(tstar, a3, color = 'orange', linestyle='dotted', linewidth=2, 
          label='$a=1000$')



plt.ylabel('Output')
plt.xlabel('Input')
plt.legend(facecolor='white', framealpha=1, edgecolor='black')
#plt.savefig('samplesPerKernel.pdf', bbox_inches='tight')
