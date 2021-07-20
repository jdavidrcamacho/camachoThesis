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

time = np.linspace(1, 10, 25)
val1 =np.random.randn(25) + 5*np.sin(2*np.pi*time/5)
val1err = 0.1*np.ones_like(val1)

mean = means.Constant(0)

tstar = np.linspace(time.min()-1, time.max()+1, 1000)

plt.rcParams['figure.figsize'] = [10/1.5, 3]
plt.figure()
plt.errorbar(time, val1, val1err, fmt='.')

eta1, eta2, eta3, eta4 = 1, 3, 5, 0.1
kernel = kernels.QuasiPeriodic(eta1, eta2, eta3, eta4)
tedibear = process.GP(kernel, mean, time, val1, val1err)
a1,s,_ = tedibear.prediction(kernel, mean, tstar)
plt.plot(tstar, a1, color='blue', linestyle='solid', linewidth=1.5, 
          label='$\eta_4 = 0.1$')

eta1, eta2, eta3, eta4 = 1, 3, 5, 1
kernel = kernels.QuasiPeriodic(eta1, eta2, eta3, eta4)
tedibear = process.GP(kernel, mean, time, val1, val1err)
a2,s,_ = tedibear.prediction(kernel, mean, tstar)
plt.plot(tstar, a2, color='red', linestyle='dashed', linewidth=1.5, 
          label='$\eta_4 = 1$')

eta1, eta2, eta3, eta4 = 1, 3, 5, 10
kernel = kernels.QuasiPeriodic(eta1, eta2, eta3, eta4)
tedibear = process.GP(kernel, mean, time, val1, val1err)
a3,s,_ = tedibear.prediction(kernel, mean, tstar)
plt.plot(tstar, a3, color = 'green', linestyle='dotted', linewidth=1.5, 
          label='$\eta_4 = 10$')
plt.ylabel('Output')
plt.xlabel('Input')
plt.legend(facecolor='white', framealpha=1, edgecolor='black')
plt.savefig('samplesPerKernel.pdf', bbox_inches='tight')
