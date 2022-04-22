import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
#np.random.seed(23011990) #to create the data used on the thesis use this seed
from astropy.timeseries import LombScargle

fig, axs = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(w=10, h=7)

def bin_data():
    time,rv,rverr,rhk,rhkerr, s,serr, \
        bis,biserr,fwhm,fwhmerr, c, cerr = np.loadtxt('CleanedSun_Dumusque.txt', 
                                                      skiprows=1, unpack=True,
                                                      usecols=(0,1,2,3,4,5,6,7,
                                                               8,9,10,11,12))
        #time = time[0:10]
        #plt.plot(time, rv, '*')

    intTime = np.int64(time) #integer values for time
    #singleTime = np.unique(intTime) + 0.5# removing repeated time
    #halfTime = list(map(truncate, time)) # truncated times to 1 decimal place
    initialTime = intTime[0] + 0.5 -1
    finalTime = intTime[-1] + 0.5 + 1
    timespan = int(finalTime-initialTime)

    newTime = []
    binRV, binRVerr = [], []
    binRHK, binRHKerr = [], []
    binS, binSerr = [], []
    binBIS, binBISerr = [], []
    binFWHM, binFWHMerr = [], []
    binC, binCerr = [], []
    i = 0
    while i < timespan:
        #for i in range(0, timespan):
            pos = np.where((time > initialTime) & (time < initialTime+1))
            if pos[0].size < 3:
                pass
            else:
                randStart = np.random.randint(0, pos[0].size - 2)
                if (time[pos[0][1]]-time[pos[0][0]]>(6/(24*60))) or (time[pos[0][2]]-time[pos[0][1]]>(6/(24*60))):
                    pass
                else:
                    #binning times
                    #newTime.append(np.mean(time[pos[0][randStart:randStart+3]]))
                    #binning RVs
                    binRV.append(np.mean(rv[pos[0][randStart:randStart+3]]))
                    #plt.plot(time[pos[0][randStart:randStart+3]] ,rv[pos[0][randStart:randStart+3]], '*')
                    binRVerr.append(np.sqrt(np.sum(rverr[pos[0][randStart:randStart+3]]**2))/3)
                    newTime.append(np.average(time[pos[0][randStart:randStart+3]], 
                                              weights=1/(rverr[pos[0][randStart:randStart+3]]**2)))
                    #plt.errorbar(time[pos[0][randStart:randStart+3]],
                    #         rv[pos[0][randStart:randStart+3]],
                    #         rverr[pos[0][randStart:randStart+3]],
                    #         fmt='*')
                    #binning RHK
                    binRHK.append(np.mean(rhk[pos[0][randStart:randStart+3]]))
                    binRHKerr.append(np.sqrt(np.sum(rhkerr[pos[0][randStart:randStart+3]]**2))/3)
                    #binning S index
                    binS.append(np.mean(s[pos[0][randStart:randStart+3]]))
                    binSerr.append(np.sqrt(np.sum(serr[pos[0][randStart:randStart+3]]**2))/3)
                    #binning BIS
                    binBIS.append(np.mean(bis[pos[0][randStart:randStart+3]]))
                    binBISerr.append(np.sqrt(np.sum(biserr[pos[0][randStart:randStart+3]]**2))/3)
                    #binning FWHM
                    binFWHM.append(np.mean(fwhm[pos[0][randStart:randStart+3]]))
                    binFWHMerr.append(np.sqrt(np.sum(fwhmerr[pos[0][randStart:randStart+3]]**2))/3)
                    #CONTRAST
                    binC.append(np.mean(c[pos[0][randStart:randStart+3]]))
                    binCerr.append(np.sqrt(np.sum(cerr[pos[0][randStart:randStart+3]]**2))/3)
            i += 1
            initialTime += 1

    results = np.stack((newTime, np.array(binRV), np.array(binRVerr), 
                        np.array(binRHK), np.array(binRHKerr), 
                        np.array(binS), np.array(binSerr), 
                        np.array(binBIS), np.array(binBISerr), 
                        np.array(binFWHM), np.array(binFWHMerr),
                        np.array(binC), np.array(binCerr)))
    results = results.T

    print('data size:', len(binRV))
    np.savetxt('sunBinned_Dumusque.txt', results, delimiter='\t',
               header ="BJD\tRV\tRVerr\tRHK\tRHKerr\tS\tSerr\tBIS\tBISerr\tFWHM\tFWHMerr\tConstrast\tContrasterr",  
               comments='')
    
    time = results[:,0]
    rvFit = np.poly1d(np.polyfit(time, results[:,1], 1))
    rv = results[:,1]-rvFit(time)
    rverr = results[:,2]
    axs[0].plot(time, rv, '.')
    f1, p1 = LombScargle(time, rv, rverr).autopower()
    axs[1].semilogx(1/f1, p1, linewidth=1)
    axs[1].set_ylabel('Power')
    axs[1].tick_params(axis='both', which='both', labelbottom=True)
    axs[1].set_xlabel('Period (days)')

for i in range(5):
    print(i)
    bin_data()
plt.show()