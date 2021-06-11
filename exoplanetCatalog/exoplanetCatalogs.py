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


###### MASS
m = data.mass.dropna() * 317.8
mSini = data.mass_sini.dropna() * 317.8
mAll = pd.concat([m,mSini])

colors = ['blue', 'red', 'green']

#bins=100**np.linspace(-0.5, 2.5, 50)
#plt.rcParams['figure.figsize'] = [15, 5]
#plt.figure()
#plt.xscale('log')
#plt.hist(mAll,  bins=bins, color=colors[0], edgecolor='navy', linewidth=1)
## plt.hist(m, bins=bins,  label='$m$', color=colors[1], alpha=0.5)
## plt.hist(msini, bins=bins, label='$m\sin i$', color=colors[2], alpha=0.5)
#plt.hist(m, bins=bins, histtype='step', stacked=True, fill=False, 
#         label='$m$', color=colors[1], linewidth=2)
#plt.hist(mSini, bins=bins, histtype='step', stacked=True, fill=False, 
#         label='$m\sin i$', color=colors[2], linewidth=2)
#plt.xlabel('Mass ($M_{Earth}$)')
#plt.ylabel('Frequency')
#plt.legend(loc='upper right')
#plt.savefig('01_planetaryMasses.pdf', bbox_inches='tight')

###### RADIUS
#r = data.radius.dropna()*11.209

#bins=100**np.linspace(-0.25, 1, 50)
#plt.rcParams['figure.figsize'] = [15, 5]
#plt.figure()
#plt.xscale('log')
#plt.hist(r,  bins=bins, color=colors[0], edgecolor='navy', linewidth=1)
#plt.xlabel('Radius ($R_{Earth}$)')
#plt.ylabel('Frequency')
#plt.savefig('02_planetaryRadius.pdf', bbox_inches='tight')

###### MASS-Period SCATTER PLOT
df = pd.DataFrame(data)
df.set_index("detection_type", inplace = True)

#plt.rcParams['figure.figsize'] = [10, 10]
#plt.figure()
#plt.yscale('log')
#plt.xscale('log')
#plt.xlabel('Period (days)')
#plt.ylabel('Mass ($M_{Jupiter}$)')
result1 = df.loc["Primary Transit"]
#plt.scatter(result1.orbital_period, result1.mass, 
#            marker='^', color='blue', alpha=0.9, label='Primary transit')

result2 = df.loc["Radial Velocity"]
#plt.scatter(result2.orbital_period, result2.mass, 
#            marker='v', color='red', alpha=0.9)
#plt.scatter(result2.orbital_period, result2.mass_sini, 
#            marker='v', color='red', alpha=0.9, label='Radial velocity')

result3 = df.loc["Imaging"]
#plt.scatter(result3.orbital_period, result3.mass, 
#            marker='o', color='grey',alpha=0.9, label='Imaging')

result4 = df.loc["Microlensing"]
#plt.scatter(result4.orbital_period, result4.mass, 
#            marker='*', color='green',alpha=0.9, label='Microlensing')

result5 = df.loc["Astrometry"]
#plt.scatter(result5.orbital_period, result5.mass, 
#            marker='P', color='purple',alpha=0.9, label='Astrometry')
#plt.legend()
#plt.savefig('03_massPeriodDistribution.pdf', bbox_inches='tight')
#plt.close('all')


###### Discoveries by Year
dates = data.discovered
first = int(data.discovered.min())
latest = int(data.discovered.max())
bins = np.linspace(first, latest, latest-first+1)

#plt.rcParams['figure.figsize'] = [10, 10]
#plt.figure()
#plt.hist(dates, bins=bins, cumulative=True, 
#         color=colors[0], edgecolor='navy', linewidth=1)
#plt.xlabel('Year of discovery')
#plt.ylabel('Cumulative Number of discoveries')
#plt.savefig('04_discoveriesCumulative.pdf', bbox_inches='tight')
#plt.close('all')


##### Combined plots
from datetime import date
today = date.today()
today = today.strftime("%d/%m/%Y")

plt.rcParams['figure.figsize'] = [10/1.5, 15/1.5]
fig, axs = plt.subplots(nrows=2, ncols=1, squeeze=True)
axs[0].hist(dates, bins=bins, cumulative=True, 
         color=colors[0], edgecolor='navy', linewidth=1)
textstr = '\n'.join(('{0} confirmed exoplanets as of {1}'.format(dates.size, today),
                     ))
props = dict(boxstyle='round', facecolor='white', alpha=1.00, edgecolor='black')
axs[0].text(0.05, 0.95, textstr, transform=axs[0].transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
axs[0].set_xlabel('Year of discovery')
axs[0].set_ylabel('Cumulative Number of discoveries')
axs[1].set_yscale('log')
axs[1].set_xscale('log')
axs[1].set_xlabel('Period (days)')
axs[1].set_ylabel('Mass ($M_{Jupiter}$)')
axs[1].scatter(result1.orbital_period, result1.mass, s=10, facecolors='none',
            marker='^', color='blue', alpha=0.9, label='Primary transit')
axs[1].scatter(result2.orbital_period, result2.mass, s=10, facecolors='none',
            marker='v', color='red', alpha=0.9)
axs[1].scatter(result2.orbital_period, result2.mass_sini, s=10, facecolors='none',
            marker='v', color='red', alpha=0.9, label='Radial velocity')
axs[1].scatter(result3.orbital_period, result3.mass, s=10, facecolors='none',
            marker='*', color='green',alpha=0.9, label='Imaging')
axs[1].scatter(result4.orbital_period, result4.mass, s=10, facecolors='none',
            marker='o', color='orange',alpha=0.9, label='Microlensing')
axs[1].scatter(result5.orbital_period, result5.mass, s=10, facecolors='none',
            marker='P', color='purple',alpha=0.9, label='Astrometry')
plt.legend(facecolor='white', framealpha=1, edgecolor='black')
plt.savefig('05_discoveriesAndMassPeriod.pdf', bbox_inches='tight')
plt.close('all')

