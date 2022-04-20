import numpy as np

data = np.loadtxt("26965_activity.csv",delimiter=',', skiprows=1, usecols=(1,4,5,11,12))
time = data[:,0].T
val1, val1err = data[:,1].T, data[:,2].T
val2, val2err = data[:,3].T, 2*val1err

##### Setting priors #####
from scipy import stats
from loguniform import ModifiedLogUniform
stats.loguniform = stats.reciprocal

#node function
neta1 = stats.loguniform(0.1, 2*np.max([val1.ptp(),val2.ptp()]))
neta2 = stats.loguniform(np.ediff1d(time).mean(), 10*time.ptp())
neta3 = stats.uniform(5, 50 -5)
neta4 = stats.loguniform(0.1, 5)

#weight function
weta1_1 = stats.loguniform(0.1, 2*val1.ptp())
weta2_1 = stats.loguniform(np.ediff1d(time).mean(), 10*time.ptp())
weta1_2 = stats.loguniform(0.1, 2*val2.ptp())
weta2_2 = stats.loguniform(np.ediff1d(time).mean(), 10*time.ptp())

#Mean function
#(1/pi)*(1/(1+slope*slope)) 
slope1 = stats.norm(0, 1)
offset1 = stats.uniform(val1.min(), val1.max() -val1.min())
slope2 = stats.norm(0, 1)
offset2 = stats.uniform(val2.min(), val2.max() -val2.min())


#Jitter
jitt1 = ModifiedLogUniform(0.1, 2*val1.ptp())
jitt2 = ModifiedLogUniform(0.1, 2*val2.ptp())



def priors():
    return np.array([neta1, neta2, neta3, neta4, 
                     weta1_1, weta2_1, weta1_2, weta2_2, 
                     slope1, offset1, slope2, offset2,
                     jitt1, jitt2])
