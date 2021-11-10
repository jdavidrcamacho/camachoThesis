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

tstar= np.linspace(0, 100, 1000)
val1 = np.ones_like(tstar)
val1err = 0.1*np.ones_like(tstar)

n1, n2, n3, n4, s = 1, 50, 20, 0.75, 0
w1, w2 = 1, 10000

sampleNum = 3
lstyle =['solid', 'dashed', 'dashdot', 'dotted',]
colors= ['blue', 'red', 'green', 'orange']
linewidth=2


# from tedi import kernels, process, means
# kernel1 = kernels.QuasiPeriodic(n1, n2, n3, n4) #+ kernels.WhiteNoise(s)
# mean = means.Constant(0)
# tedibear0 = process.GP(kernel1, mean, tstar, val1, val1err)
# a0 = tedibear0.sample(kernel1, tstar)

# fig, axs = plt.subplots(nrows=1,ncols=1, sharex=True)
# fig.set_size_inches(w=7, h=2)
# for i in range(sampleNum):
#     a0 = tedibear0.sample(kernel1, tstar)
#     axs.plot(tstar, a0, linewidth=linewidth, linestyle=lstyle[i], color=colors[i])
# axs.set_ylabel('Output (m/s)')
# axs.set_xlabel('Input (days)')
# fig.savefig('samples_of3GPs.pdf', bbox_inches='tight')

from gprn.covFunction import SquaredExponential, QuasiPeriodic, Periodic
from gprn.meanFunction import Linear, Constant
from gprn.meanField import inference

# GPRN = inference(1, tstar, val1, val1err)
# node = [Periodic(n1, n3, n4)]
# weight = [SquaredExponential(w1, n2)]
# mean = [Constant(0)]
# jitter = [s]

# fig, axs = plt.subplots(nrows=1,ncols=1, sharex=True)
# fig.set_size_inches(w=7, h=2)
# for i in range(sampleNum):
#     samples1 = GPRN.sampleIt(node[0], time=tstar)
#     samples2 = GPRN.sampleIt(weight[0], time=tstar)
#     axs.plot(tstar, samples1*samples2, linewidth=linewidth, linestyle=lstyle[i], color=colors[i])
# axs.set_ylabel('Output (m/s)')
# axs.set_xlabel('Input (days)')
# fig.savefig('samples_of3GPRN_periodic.pdf', bbox_inches='tight')

GPRN = inference(1, tstar, val1, val1err)
node = [QuasiPeriodic(n1, n2, n3, n4)]
weight = [SquaredExponential(w1,w2)]
mean = [Constant(0)]
jitter = [s]

fig, axs = plt.subplots(nrows=1,ncols=1, sharex=True)
fig.set_size_inches(w=7, h=2)
for i in range(sampleNum):
    samples1 = GPRN.sampleIt(node[0], time=tstar)
    samples2 = GPRN.sampleIt(weight[0], time=tstar)
    axs.plot(tstar, samples1*samples2, linewidth=linewidth, linestyle=lstyle[i], color=colors[i])
axs.set_ylabel('Output (m/s)')
axs.set_xlabel('Input (days)')
fig.savefig('samples_of3GPRN_quasiperiodic.pdf', bbox_inches='tight')