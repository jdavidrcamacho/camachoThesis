import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')


linesize = 1

################################################################################
rotP10700 = 34
data = np.loadtxt("10700_activity.csv",delimiter=',', skiprows=1, usecols=(1,4,5,11,12))
time1 = data[:,0].T #in MJD
val1RV, val1RVerr = data[:,1].T, data[:,2].T
val1FW, val1FWerr = data[:,3].T, 2*val1RVerr

plt.figure()
plt.title('HD 10700 observations')
plt.errorbar(time1, val1RV, val1RVerr, fmt='.')

#Lowell Observatory is on Flagstaff, Arizona, United States.
# UTC Offset: UTC -7; 8 hours behind Porto
import datetime as dt
from tedi import time

jdTime = time.mjd_to_jd(time1)

normalTime = []
for i, j in enumerate(jdTime):
    normalTime.append(time.jd_to_datetime(j))
normalTime = np.array(normalTime) #Greenwich time

delta = dt.timedelta(hours=7)
expresTime = []
for i, j in enumerate(normalTime):
    expresTime.append(np.array([i, j - delta]))
expresTime = np.array(expresTime) # Arizona Time

#variations
variations = [val1RV[0: 2+1].ptp(), val1RV[3: 4+1].ptp(), val1RV[5: 7+1].ptp(),
              val1RV[8: 10+1].ptp(), val1RV[11: 15+1].ptp(), val1RV[16: 19+1].ptp(),
              val1RV[20: 25+1].ptp(),val1RV[26: 45+1].ptp(),val1RV[46: 50+1].ptp(),
              val1RV[51: 74+1].ptp(),val1RV[75: 79+1].ptp(),val1RV[80: 84+1].ptp(),
              val1RV[85: 89+1].ptp(),val1RV[90: 94+1].ptp(),val1RV[95: 99+1].ptp(),
              val1RV[100: 104+1].ptp(),val1RV[105: 109+1].ptp(),val1RV[110: 113+1].ptp(),
              val1RV[114: 117+1].ptp(),val1RV[118: 120+1].ptp(),val1RV[121: 123+1].ptp(),
              val1RV[124: 126+1].ptp(),val1RV[127: 129+1].ptp(),val1RV[130: 132+1].ptp(),
              val1RV[133: 135+1].ptp(),val1RV[136: 140+1].ptp(),val1RV[141: 145+1].ptp(),
              val1RV[146].ptp(),val1RV[147: 150+1].ptp(),val1RV[151: 155+1].ptp(),
              val1RV[156: 160+1].ptp(),val1RV[161: 163+1].ptp(),val1RV[164: 168+1].ptp(),
              val1RV[169: 173+1].ptp()]
variations = np.array(variations)
plt.figure()
plt.title('HD 10700 RV variation per observation (average = {0}m/s)'.format(np.round(variations.mean(),3)))
plt.plot(variations, '.k')