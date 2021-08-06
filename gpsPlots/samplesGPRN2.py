import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')

tstar= np.linspace(0, 100, 1000)
val1 = np.ones_like(tstar)
val1err = 0.1*np.ones_like(tstar)

s=0
lstyle =['solid', 'dashed', 'dotted', 'dashdot']
colors= ['blue', 'red', 'green', 'orange']
linewidth=2

fig = plt.figure(constrained_layout=True, figsize=(6, 8))
axs = fig.subplot_mosaic(
    [
        ['node1','predictive1'],
        ['weight1', 'predictive1'],
        ['node2', 'predictive2'],
        ['weight2', 'predictive2'],
        ['node3', 'predictive3'],
        ['weight3', 'predictive3'],
    ],
)

from gprn.covFunction import SquaredExponential, QuasiPeriodic, Periodic
from gprn.meanFunction import Linear, Constant
from gprn.meanField import inference

GPRN = inference(1, tstar, val1, val1err)
node = [QuasiPeriodic(1, 50, 20, 0.75)]
weight = [SquaredExponential(1, 10)]
mean = [Constant(0)]
jitter = [s]
samples1 = GPRN.sampleIt(node[0], time=tstar)
samples2 = GPRN.sampleIt(weight[0], time=tstar)
axs['predictive1'].set(xlabel='Input', ylabel='GPRN output')
axs['predictive1'].plot(tstar, samples1*samples2, linewidth=linewidth, 
                        linestyle=lstyle[0], color=colors[0])
axs['node1'].set(ylabel='Node output')
axs['node1'].plot(tstar, samples1, linewidth=linewidth, linestyle=lstyle[0], 
                  color=colors[0])
axs['weight1'].set(xlabel='Input', ylabel='Weight output')
axs['weight1'].plot(tstar, samples2, linewidth=linewidth, linestyle=lstyle[0], 
                    color=colors[0])

GPRN = inference(1, tstar, val1, val1err)
node = [QuasiPeriodic(1, 50, 20, 0.75)]
weight = [SquaredExponential(1, 100)]
mean = [Constant(0)]
jitter = [s]
# samples1 = GPRN.sampleIt(node[0], time=tstar)
samples2 = GPRN.sampleIt(weight[0], time=tstar)
axs['predictive2'].set(xlabel='Input', ylabel='GPRN output')
axs['predictive2'].plot(tstar, samples1*samples2, linewidth=linewidth, 
                        linestyle=lstyle[1], color=colors[1])
axs['node2'].set(ylabel='Node output')
axs['node2'].plot(tstar, samples1, linewidth=linewidth, linestyle=lstyle[1], 
                  color=colors[1])
axs['weight2'].set(xlabel='Input', ylabel='Weight output')
axs['weight2'].plot(tstar, samples2, linewidth=linewidth, linestyle=lstyle[1], 
                    color=colors[1])

GPRN = inference(1, tstar, val1, val1err)
node = [QuasiPeriodic(1, 50, 20, 0.75)]
weight = [SquaredExponential(1, 1000)]
mean = [Constant(0)]
jitter = [s]
# samples1 = GPRN.sampleIt(node[0], time=tstar)
samples2 = GPRN.sampleIt(weight[0], time=tstar)
axs['predictive3'].set(xlabel='Input', ylabel='GPRN output')
axs['predictive3'].plot(tstar, samples1*samples2, linewidth=linewidth, 
                        linestyle=lstyle[2], color=colors[2])
axs['node3'].set(ylabel='Node output')
axs['node3'].plot(tstar, samples1, linewidth=linewidth, linestyle=lstyle[2], 
                  color=colors[2])
axs['weight3'].set(xlabel='Input', ylabel='Weight output')
axs['weight3'].plot(tstar, samples2, linewidth=linewidth, linestyle=lstyle[2], 
                    color=colors[2])
fig.savefig('samplesGPRN2.pdf', bbox_inches='tight')
# plt.close('all')
