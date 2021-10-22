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
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.linewidth'] = 2
plt.close('all')

###### Data .rdb file #####
time, sunspot = np.loadtxt("SN_m_tot_V2.0.txt", skiprows = 0, 
                           unpack = True, usecols = (2, 3))

plt.rcParams['figure.figsize'] = [0.7*10, 0.6*5]
plt.figure()
plt.plot(time, sunspot, '.', color='blue')
plt.xlabel('Year')
plt.ylabel('Sunsplot number')
plt.tight_layout(h_pad=0.7, w_pad=0.7)
plt.savefig('sunspotsNumber.pdf', bbox_inches='tight')
# plt.close('all')
