import numpy as np
import matplotlib
#matplotlib.rcParams['figure.figsize'] = [15, 2*5]
#matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})

import matplotlib.pyplot as plt
plt.close('all')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.linewidth'] = 2
from matplotlib.ticker import AutoMinorLocator
###### Data .rdb file #####
time, sunspot = np.loadtxt("SN_d_tot_V2.0.txt", skiprows = 0, 
                           unpack = True, usecols = (3, 4))

values = np.where((time > 2015) & (time < 2019))
time = time[values[0]]
sunspot = sunspot[values[0]]
values1 = np.where((sunspot==0))
values2 = np.where((sunspot!=0))

values = np.where((time > 2015) & (time < 2020))
time = time[values[0]]
sunspot = sunspot[values[0]]
# values = np.where((sunspot >= 0))
# time = time[values[0]]
# sunspot = sunspot[values[0]]

plt.rcParams['figure.figsize'] = [0.7*10, 0.6*5]
plt.figure()
plt.plot(time, sunspot, '.', color='black', alpha=0.5)
plt.xlabel('Year')
plt.ylabel('Sunsplot number')
plt.tight_layout(h_pad=0.7, w_pad=0.7)
plt.grid()
plt.savefig('daySunspotsNumber.pdf', bbox_inches='tight')
# plt.close('all')
