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
colors = ['blue', 'red', 'green', 'orange', 'purple']

plt.close('all')

import pandas as pd
data = pd.read_csv('exoplanet.eu_catalog.csv')

###### MASS
m = data.mass.dropna() * 317.8
mSini = data.mass_sini.dropna() * 317.8
mAll = pd.concat([m,mSini])

###### MASS-Period SCATTER PLOT
df = pd.DataFrame(data)
df.set_index("detection_type", inplace = True)

result1 = df.loc["Primary Transit"]
result2 = df.loc["Radial Velocity"]
result3 = df.loc["Imaging"]
result4 = df.loc["Microlensing"]
result5 = df.loc["Astrometry"]

###### Discoveries by Year
dates = data.discovered

from datetime import date
today = date.today()
today = today.strftime("%d/%m/%Y")

plt.rcParams['figure.figsize'] = [4, 4]
fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=True)

textstr = '\n'.join(('{0} confirmed exoplanets as of {1}'.format(dates.size, today),
                     ))
props = dict(boxstyle='round', facecolor='white', alpha=1.00, edgecolor='black')

axs.set_yscale('log')
axs.set_xscale('log')
axs.set_xlabel('Period (days)')
axs.set_ylabel('Mass ($M_{Jupiter}$)')
axs.scatter(result1.orbital_period, result1.mass, s=7.5, facecolors='none',
            marker='^', color=colors[0], alpha=0.85, label='Primary transit')
axs.scatter(result2.orbital_period, result2.mass, s=7.5, facecolors='none',
            marker='v', color=colors[1], alpha=0.82)
axs.scatter(result2.orbital_period, result2.mass_sini, s=7.5, facecolors='none',
            marker='v', color=colors[1], alpha=0.85, label='Radial velocity')
axs.scatter(result3.orbital_period, result3.mass, s=7.5, facecolors='none',
            marker='*', color=colors[2],alpha=0.85, label='Imaging')
axs.scatter(result4.orbital_period, result4.mass, s=7.5, facecolors='none',
            marker='o', color=colors[3],alpha=0.85, label='Microlensing')
axs.scatter(result5.orbital_period, result5.mass, s=7.5, facecolors='none',
            marker='P', color=colors[4],alpha=0.85, label='Astrometry')
plt.legend(facecolor='whitesmoke', framealpha=1, edgecolor='black',
           loc='lower right')
plt.savefig('05_discoveriesAndMassPeriod.pdf', bbox_inches='tight')
# plt.close('all')

