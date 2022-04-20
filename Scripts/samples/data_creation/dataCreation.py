import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')

from gpyrn import meanfield, covfunc, meanfunc

time = np.array([1])

seed = np.random.randint(0,100000)
print(seed)
np.random.seed(66884)

i = 1
while i < 50:
    t = np.random.randint(1, 150, 1)[0]
    if t not in time:
        time = np.append(time, t)
        i += 1
time = np.sort(time)

val1 = np.ones_like(time) 
err = np.random.normal(0, 1, 50)
val1err = err.std() * np.ones_like(time)

GPRN = meanfield.inference(1, time, val1, val1err, val1, val1err, val1, val1err)

node = [covfunc.QuasiPeriodic(11, 17, 23, 0.75)]
weigths =[covfunc.SquaredExponential(1, 25),
          covfunc.SquaredExponential(1, 50),
          covfunc.SquaredExponential(1, 100)]
means = [meanfunc.Constant(0), meanfunc.Constant(0), meanfunc.Constant(0)]
jitter = [0.1, 0.1, 0.1]

n = GPRN.sampleIt(node[0], time)
w1 = GPRN.sampleIt(weigths[0], time)
w2 = GPRN.sampleIt(weigths[1], time)
w3 = GPRN.sampleIt(weigths[2], time)

results = np.stack((np.array(time), 
                    np.array(n), np.array(w1), 
                    np.array(w2), np.array(w3)))
results = results.T

np.savetxt('componentsPoints.txt', results, delimiter='\t',
           header ="time\tn\tw1\tw2\tw3",  
           comments='')

nw1 = n*w1 + err
nw2 = n*w2 + err
nw3 = n*w3 + err
fig = plt.figure(constrained_layout=True, figsize=(14, 7))
axs = fig.subplot_mosaic( [['X', 'weight1','serie1'],
                           ['node', 'weight1','serie1'],
                           ['node', 'weight2','serie2'],
                           ['node', 'weight2','serie2'],
                           ['node', 'weight3', 'serie3'],
                           ['X', 'weight3', 'serie3'],],
                         empty_sentinel="X",)

axs['node'].set_title('Node')
axs['node'].set_xlabel('Time (days)')
axs['node'].plot(time, n, '.-', color='blue')
axs['weight1'].set_title('Weights')
axs['weight1'].plot(time, w1, '.-', color='red')
axs['weight1'].set_ylabel('1st weight (m/s)')
axs['weight1'].tick_params(axis='both', which='both', labelbottom=False)
axs['weight2'].plot(time, w2, '.-', color='gold')
axs['weight2'].set_ylabel('2nd weight (m/s)')
axs['weight2'].tick_params(axis='both', which='both', labelbottom=False)
axs['weight3'].plot(time, w3, '.-', color='silver')
axs['weight3'].set_xlabel('Time (days)')
axs['weight3'].set_ylabel('3rd weight (m/s)')
axs['serie1'].set_title('Time series')
axs['serie1'].set_ylabel('Serie 1 (m/s)')
axs['serie1'].errorbar(time, nw1, val1err, 
                       fmt='.-', color='purple')
axs['serie1'].tick_params(axis='both', which='both', labelbottom=False)
axs['serie2'].set_ylabel('Serie 2 (m/s)')
axs['serie2'].errorbar(time, nw2, val1err, 
                       fmt='.-', color='green')
axs['serie2'].tick_params(axis='both', which='both', labelbottom=False)
axs['serie3'].set_ylabel('Serie 3 (m/s)')
axs['serie3'].errorbar(time, nw3, val1err, 
                       fmt='.-', color='cornflowerblue')
axs['serie3'].set_xlabel('Time (days)')
plt.tight_layout()
fig.savefig('data.pdf', bbox_inches='tight')
plt.close('all')

results = np.stack((np.array(time), 
                    np.array(nw1), np.array(val1err), 
                    np.array(nw2), np.array(val1err), 
                    np.array(nw3), np.array(val1err)))
results = results.T

np.savetxt('sample50points.txt', results, delimiter='\t',
           header ="time\tRV\tRVerr\tBIS\tBISerr\tFWHM\tFWHMerr",  
           comments='')

time,rv,rverr,bis,biserr,fw,fwerr = np.loadtxt("sample50points.txt", skiprows = 1, 
                                               unpack = True, 
                                               usecols = (0,1,2,3,4,5,6))
time = time


# Trend removal ################################################################
rvFit = np.poly1d(np.polyfit(time, rv, 1))
rv = np.array(rv)-rvFit(time)

bisFit = np.poly1d(np.polyfit(time, bis, 1))
bis = np.array(bis)-bisFit(time)

fwFit = np.poly1d(np.polyfit(time, fw, 1))
fw = np.array(fw)-fwFit(time)


fig, axs = plt.subplots(nrows=3, ncols=2)
fig.set_size_inches(w=14, h=7)

axs[0,0].errorbar(time, rv, rverr, fmt= '.b')
axs[0,0].tick_params(axis='both', which='both', labelbottom=False)
axs[0,0].set_ylabel('Serie 1 (m/s)')

axs[1,0].errorbar(time, bis, biserr, fmt= '.b')
axs[1,0].set_ylabel('Serie 2(m/s)')
axs[1,0].tick_params(axis='both', which='both', labelbottom=False)

axs[2,0].errorbar(time, fw, fwerr, fmt= '.b')
axs[2,0].set_ylabel('Serie 3 (m/s)')
axs[2,0].tick_params(axis='both', which='both', labelbottom=True)
axs[2,0].set_xlabel('Time (days)')

from astropy.timeseries import LombScargle
linesize = 1


f1, p1 = LombScargle(time, rv, rverr).autopower()
axs[0,1].semilogx(1/f1, p1, color='blue', linewidth=linesize)
bestf = f1[np.argmax(p1)]
bestp = 1/bestf
axs[0,1].axvline(x=23, ymin=p1.min(), ymax=100*p1.max(), color='red', alpha=1,
               linewidth=linesize)
axs[0,1].tick_params(axis='both', which='both', labelbottom=False)

#false alarm
falseAlarms1 = LombScargle(time, rv, rverr).false_alarm_level([0.1,0.01,0.001])
one = falseAlarms1[1] * np.ones_like(f1)
axs[0,1].plot(1/f1, one, color='red', linestyle='dashed', alpha=1,linewidth=linesize)
axs[0,1].set_xlim(1,time.ptp())

f2, p2 = LombScargle(time, bis, biserr).autopower()
axs[1,1].semilogx(1/f2, p2, color='blue', linewidth=linesize)
axs[1,1].set_ylabel('Normalized power')
bestf = f2[np.argmax(p2)]
bestp = 1/bestf
axs[1,1].axvline(x=23, ymin=p2.min(), ymax=100*p2.max(), color='red', alpha=1,
               linewidth=linesize)
axs[1,1].tick_params(axis='both', which='both', labelbottom=False)
#false alarm
falseAlarms2 = LombScargle(time, bis, biserr).false_alarm_level([0.1,0.01,0.001])
one = falseAlarms2[1] * np.ones_like(f2)
axs[1,1].plot(1/f2, one, color='r', linestyle='dashed', alpha=1, linewidth=linesize)
axs[1,1].set_xlim(1,time.ptp())

f3, p3 = LombScargle(time, fw, fwerr).autopower()
falseAlarms3 = LombScargle(time, fw, fwerr).false_alarm_level([0.1,0.01,0.001])
axs[2,1].semilogx(1/f3, p3, color='blue', linewidth=linesize)
bestf = f3[np.argmax(p3)]
bestp = 1/bestf
axs[2,1].axvline(x=23, ymin=p3.min(), ymax=100*p3.max(), color='r', alpha=1,
               linewidth=linesize)
axs[2,1].tick_params(axis='both', which='both', labelbottom=True)
falseAlarms3 = LombScargle(time, fw, fwerr).false_alarm_level([0.1,0.01,0.001])
one = falseAlarms3[1] * np.ones_like(f3)
axs[2,1].plot(1/f3, one, color='red', linestyle='dashed', alpha=1, linewidth=linesize)
axs[2,1].set_xlim(1,time.ptp())
axs[2,1].set_xlabel('Period (days)')

plt.savefig('periodograms.pdf', bbox_inches='tight')
