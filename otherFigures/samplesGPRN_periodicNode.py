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

n1, n2, n3, n4, s = 1, 53, 19, 0.75, 0.1
w1, w2 = 1, 10000

sampleNum = 3
lstyle =['solid', 'dashed', 'dotted', 'dashdot']
colors= ['blue', 'red', 'green', 'orange']
linewidth=2

fig, axs = plt.subplots(nrows=3,ncols=1, sharex=True)
fig.set_size_inches(w=5, h=10)

from gprn.covFunction import SquaredExponential, QuasiPeriodic, Periodic
from gprn.meanFunction import Linear, Constant
from gprn.completeMeanField2 import inference

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
axs[0].title.set_text('GPRN with periodic node')
axs[0].set_ylabel('Output')

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
axs[1].title.set_text('GPRN with quasi-periodic node')
axs[1].set_ylabel('Output')


from tedi import kernels, process, means
kernel1 = kernels.QuasiPeriodic(n1, n2, n3, n4) #+ kernels.WhiteNoise(s)
mean = means.Constant(0)

tedibear0 = process.GP(kernel1, mean, tstar, val1, val1err)
for i in range(sampleNum):
    a0 = tedibear0.sample(kernel1, tstar)
    axs[2].plot(tstar, a0, linewidth=linewidth, linestyle=lstyle[i], color=colors[i])
axs[2].title.set_text('GP')
axs[2].set_ylabel('Output')
axs[2].set_xlabel('Input')
fig.savefig('samplesGPRN1.pdf', bbox_inches='tight')

