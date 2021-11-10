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
tstar= np.linspace(0, 100, 1000)

plt.rcParams['figure.figsize'] = [7, 12]
fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True)

# np.random.seed(1)
kernel1 = kernels.RationalQuadratic(1, 1, 1)
tedibear1 = process.GP(kernel1, mean, time, val1, val1err)
a1 = tedibear1.sample(kernel1, tstar)
axs[0].plot(tstar, a1, color='blue', linestyle='solid', linewidth=2, 
         label='$\eta_2 = 1, \\alpha=1$')

kernel2 = kernels.RationalQuadratic(1, 1, 10)
tedibear2 = process.GP(kernel2, mean, time, val1, val1err)
a2 = tedibear2.sample(kernel2, tstar)
axs[0].plot(tstar, a2, color='red', linestyle='dashed', linewidth=2, 
         label='$\eta_2 = 10, \\alpha=1$')

kernel3 = kernels.RationalQuadratic(1, 1, 100)
tedibear3 = process.GP(kernel3, mean, time, val1, val1err)
a3 = tedibear3.sample(kernel3, tstar)
axs[0].plot(tstar, a3, color = 'green', linestyle='dotted', linewidth=2, 
         label='$\eta_2 = 100, \\alpha=1$')

kernel4 = kernels.RationalQuadratic(1, 1, 1000)
tedibear4 = process.GP(kernel4, mean, time, val1, val1err)
a4 = tedibear4.sample(kernel4, tstar)
axs[0].plot(tstar, a4, color = 'orange', linestyle='dotted', linewidth=2, 
         label='$\eta_2 = 1000, \\alpha=1$')
axs[0].set_ylabel('Output')
axs[0].legend(loc='upper left', facecolor='whitesmoke', 
              framealpha=1, edgecolor='black')

# np.random.seed(1)
kernel1 = kernels.RationalQuadratic(1, 10, 1)
tedibear1 = process.GP(kernel1, mean, time, val1, val1err)
a1 = tedibear1.sample(kernel1, tstar)
axs[1].plot(tstar, a1, color='blue', linestyle='solid', linewidth=2, 
         label='$\eta_2 = 1, \\alpha = 10$')

kernel2 = kernels.RationalQuadratic(1, 10, 10)
tedibear2 = process.GP(kernel2, mean, time, val1, val1err)
a2 = tedibear2.sample(kernel2, tstar)
axs[1].plot(tstar, a2, color='red', linestyle='dashed', linewidth=2, 
         label='$\eta_2 = 10, \\alpha = 10$')

kernel3 = kernels.RationalQuadratic(1, 10, 100)
tedibear3 = process.GP(kernel3, mean, time, val1, val1err)
a3 = tedibear3.sample(kernel3, tstar)
axs[1].plot(tstar, a3, color = 'green', linestyle='dotted', linewidth=2, 
         label='$\eta_2 = 100, \\alpha = 10$')

kernel4 = kernels.RationalQuadratic(1, 10, 1000)
tedibear4 = process.GP(kernel4, mean, time, val1, val1err)
a4 = tedibear4.sample(kernel4, tstar)
axs[1].plot(tstar, a4, color = 'orange', linestyle='dotted', linewidth=2, 
         label='$\eta_2 = 1000, \\alpha=10$')
axs[1].set_ylabel('Output')
axs[1].legend(loc='upper left', facecolor='whitesmoke', 
              framealpha=1, edgecolor='black')

# np.random.seed(1)
kernel1 = kernels.RationalQuadratic(1, 100, 1)
tedibear1 = process.GP(kernel1, mean, time, val1, val1err)
a1 = tedibear1.sample(kernel1, tstar)
axs[2].plot(tstar, a1, color='blue', linestyle='solid', linewidth=2, 
         label='$\eta_2 = 1, \\alpha = 100$')

kernel2 = kernels.RationalQuadratic(1, 100, 10)
tedibear2 = process.GP(kernel2, mean, time, val1, val1err)
a2 = tedibear2.sample(kernel2, tstar)
axs[2].plot(tstar, a2, color='red', linestyle='dashed', linewidth=2, 
         label='$\eta_2 = 10, \\alpha = 100$')

kernel3 = kernels.RationalQuadratic(1, 100, 100)
tedibear3 = process.GP(kernel3, mean, time, val1, val1err)
a3 = tedibear3.sample(kernel3, tstar)
axs[2].plot(tstar, a3, color = 'green', linestyle='dotted', linewidth=2, 
         label='$\eta_2 = 100, \\alpha = 100$')

kernel4 = kernels.RationalQuadratic(1, 100, 1000)
tedibear4 = process.GP(kernel4, mean, time, val1, val1err)
a4 = tedibear4.sample(kernel4, tstar)
axs[2].plot(tstar, a4, color = 'orange', linestyle='dotted', linewidth=2, 
         label='$\eta_2 = 1000, \\alpha=100$')
axs[2].set_ylabel('Output')
axs[2].legend(loc='upper left', facecolor='whitesmoke', 
              framealpha=1, edgecolor='black')

kernel1 = kernels.RationalQuadratic(1, 1000, 1)
tedibear1 = process.GP(kernel1, mean, time, val1, val1err)
a1 = tedibear1.sample(kernel1, tstar)
axs[3].plot(tstar, a1, color='blue', linestyle='solid', linewidth=2, 
         label='$\eta_2 = 1, \\alpha = 1000$')

kernel2 = kernels.RationalQuadratic(1, 1000, 10)
tedibear2 = process.GP(kernel2, mean, time, val1, val1err)
a2 = tedibear2.sample(kernel2, tstar)
axs[3].plot(tstar, a2, color='red', linestyle='dashed', linewidth=2, 
         label='$\eta_2 = 10, \\alpha = 1000$')

kernel3 = kernels.RationalQuadratic(1, 1000, 100)
tedibear3 = process.GP(kernel3, mean, time, val1, val1err)
a3 = tedibear3.sample(kernel3, tstar)
axs[3].plot(tstar, a3, color = 'green', linestyle='dotted', linewidth=2, 
         label='$\eta_2 = 100, \\alpha = 1000$')

kernel4 = kernels.RationalQuadratic(1, 1000, 1000)
tedibear4 = process.GP(kernel4, mean, time, val1, val1err)
a4 = tedibear4.sample(kernel4, tstar)
axs[3].plot(tstar, a4, color = 'orange', linestyle='dotted', linewidth=2, 
         label='$\eta_2 = 1000, \\alpha=1000$')
axs[3].set_ylabel('Output')
axs[3].legend(loc='upper left', facecolor='whitesmoke', 
              framealpha=1, edgecolor='black')

axs[3].set_xlabel('Input')
plt.savefig('samplesRQKernel.pdf', bbox_inches='tight')
