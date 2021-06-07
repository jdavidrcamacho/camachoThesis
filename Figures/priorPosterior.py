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

kernel = kernels.SquaredExponential(1, 1)
mean = means.Constant(0)

tedibear = process.GP(kernel, mean, time, val1, val1err)

tstar= np.linspace(0, 8, 100)
plt.rcParams['figure.figsize'] = [0.7*10, 0.7*5]
fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True)

#10, 25, 58
np.random.seed(58)
for i in range(1):
    a = tedibear.sample(kernel, tstar)
    axs[0].plot(tstar, a, 'b',linewidth=2)
    axs[0].set_ylabel('Output')
    
    b = tedibear.posteriorSample(kernel,mean,val1,tstar)
    axs[1].plot(tstar, b, 'b',linewidth=2)
np.random.seed(10)
for i in range(1):
    a = tedibear.sample(kernel, tstar)
    axs[0].plot(tstar, a, '--r',linewidth=2)
    axs[0].set_ylabel('Output')
    
    b = tedibear.posteriorSample(kernel,mean,val1,tstar)
    axs[1].plot(tstar, b, '--r',linewidth=2)
np.random.seed(25)
for i in range(1):
    a = tedibear.sample(kernel, tstar)
    axs[0].plot(tstar, a, '-.g',linewidth=2)
    axs[0].set_ylabel('Output')
    
    b = tedibear.posteriorSample(kernel,mean,val1,tstar)
    axs[1].plot(tstar, b, '-.g',linewidth=2)
    
    
    
axs[1].plot(time, val1, 'ko', markersize=8)
axs[0].set_xlabel('Input')
axs[0].set_title("Prior")
axs[1].set_xlabel('Input')
axs[1].set_title("Posterior")
plt.savefig('PriorPosterior.pdf', bbox_inches='tight')

