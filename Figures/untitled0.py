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

n1, n2, n3, n4, s = 1, 50, 20, 1, 0.1
w1, w2 = 1, 10000

sampleNum = 3
lstyle =['solid', 'dashed', 'dotted', 'dashdot']
colors= ['blue', 'red', 'green', 'orange']
linewidth=2

fig, axs = plt.subplots(nrows=1,ncols=2, sharex=True)
fig.set_size_inches(w=15, h=5)

from gprn.covFunction import SquaredExponential, QuasiPeriodic, Periodic
from gprn.meanFunction import Linear, Constant
from gprn.meanField import inference

GPRN = inference(1, tstar, val1, val1err)
node = [Periodic(n1, n3, n4)]
weight = [SquaredExponential(w1, n2)]
mean = [Constant(0)]
jitter = [s]

for i in range(sampleNum):
    samples1 = GPRN.sampleIt(node[0], time=tstar)
    samples2 = GPRN.sampleIt(weight[0], time=tstar)
    # axs[0].plot(tstar, samples1, color = 'blue', linewidth=2, linestyle='dashed')
    # axs[0].plot(tstar, samples2, color = 'blue', linewidth=2, linestyle='dotted')
    axs[0].plot(tstar, samples1*samples2, linewidth=linewidth, linestyle=lstyle[i], color=colors[i])
axs[0].title.set_text('GPRN - periodic node')
axs[0].set_ylabel('Output')
axs[0].set_xlabel('Input')

GPRN = inference(1, tstar, val1, val1err)
node = [QuasiPeriodic(n1, n2, n3, n4)]
weight = [SquaredExponential(w1,w2)]
mean = [Constant(0)]
jitter = [s]

for i in range(sampleNum):
    samples1 = GPRN.sampleIt(node[0], time=tstar)
    samples2 = GPRN.sampleIt(weight[0], time=tstar)
    # axs[0].plot(tstar, samples1, color = 'blue', linewidth=2, linestyle='dashed')
    # axs[0].plot(tstar, samples2, color = 'blue', linewidth=2, linestyle='dotted')
    axs[1].plot(tstar, samples1*samples2, linewidth=linewidth, linestyle=lstyle[i], color=colors[i])
axs[1].title.set_text('GPRN - quasiperiodic node')
axs[1].set_ylabel('Output')
axs[1].set_xlabel('Input')
