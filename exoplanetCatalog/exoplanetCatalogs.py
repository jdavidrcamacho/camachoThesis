import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False})
import matplotlib.pylab as plt
plt.close('all')


import pandas as pd
data = pd.read_csv('exoplanet.eu_catalog.csv')

##### MASS
m = data.mass.dropna()
mSini = data.mass_sini.dropna()
mAll = pd.concat([m,mSini])

colors = ['blue', 'red', 'green']

bins=100**np.linspace(-1.5, 1, 50)
plt.rcParams['figure.figsize'] = [15, 5]
plt.figure()
plt.xscale('log')
plt.hist(mAll,  bins=bins, color=colors[0], edgecolor='navy', linewidth=1)
# plt.hist(m, bins=bins,  label='$m$', color=colors[1], alpha=0.5)
# plt.hist(msini, bins=bins, label='$m\sin i$', color=colors[2], alpha=0.5)
plt.hist(m, bins=bins, histtype='step', stacked=True, fill=False, 
         label='$m$', color=colors[1], linewidth=2)
plt.hist(mSini, bins=bins, histtype='step', stacked=True, fill=False, 
         label='$m\sin i$', color=colors[2], linewidth=2)
plt.xlabel('Mass ($M_{Jupiter}$)')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.savefig('01_planetaryMasses.pdf', bbox_inches='tight')

##### RADIUS
r = data.radius.dropna()

bins=100**np.linspace(-1, 0.5, 50)
plt.rcParams['figure.figsize'] = [15, 5]
plt.figure()
plt.xscale('log')
plt.hist(r,  bins=bins, color=colors[0], edgecolor='navy', linewidth=1)
plt.xlabel('Radius ($R_{Jupiter}$)')
plt.ylabel('Frequency')
plt.savefig('02_planetaryRadius.pdf', bbox_inches='tight')

##### MASS-RADIUS SCATTER PLOT
df = pd.DataFrame(data)
df.set_index("detection_type", inplace = True)

plt.rcParams['figure.figsize'] = [10, 10]
plt.figure()
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Period (days)')
plt.ylabel('Radius ($R_{Jupiter}$)')
result1 = df.loc["Primary Transit"]
plt.scatter(result1.orbital_period, result1.mass, 
            marker='^', color='blue', alpha=0.9, label='Primary transit')

result2 = df.loc["Radial Velocity"]
plt.scatter(result2.orbital_period, result2.mass, 
            marker='v', color='red', alpha=0.9)
plt.scatter(result2.orbital_period, result2.mass_sini, 
            marker='v', color='red', alpha=0.9, label='Radial velocity')

result3 = df.loc["Imaging"]
plt.scatter(result3.orbital_period, result3.mass, 
            marker='o', color='grey',alpha=0.9, label='Imaging')

result4 = df.loc["Microlensing"]
plt.scatter(result4.orbital_period, result4.mass, 
            marker='*', color='green',alpha=0.9, label='Microlensing')

result5 = df.loc["Astrometry"]
plt.scatter(result5.orbital_period, result5.mass, 
            marker='P', color='purple',alpha=0.9, label='Astrometry')
plt.legend()
plt.savefig('03_massPeriodDistribution.pdf', bbox_inches='tight')
plt.close('all')