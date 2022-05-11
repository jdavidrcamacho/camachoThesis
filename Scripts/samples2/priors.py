import numpy as np
from scipy import stats
from loguniform import ModifiedLogUniform
stats.loguniform = stats.reciprocal

def priors(filename, RVs=True, BIS=False, FWHM=False):
    time,rv,bis,fw= np.loadtxt(filename, skiprows = 1, unpack = True,
                                   usecols = (0,1,3,5))
    val1 = rv; val2 = bis; val3 = fw; 
    
    ### node function
    neta1MIN, neta1MAX = [], []
    if RVs:
        neta1MIN.append(np.std(val1))
        neta1MAX.append(2*val1.ptp())
    if BIS:
        neta1MIN.append(np.std(val2))
        neta1MAX.append(2*val2.ptp())
    if FWHM:
        neta1MIN.append(np.std(val3))
        neta1MAX.append(2*val3.ptp())

    neta1 = ModifiedLogUniform(np.min(neta1MIN), np.max(neta1MAX))
    neta2 = stats.loguniform(np.ediff1d(time).mean(), 10*time.ptp())
    neta3 = stats.uniform(10, 50 -10)
    neta4 = stats.loguniform(0.1, 5)
    priorArray = [neta1, neta2, neta3, neta4] 

    ### weight function
    weta1_1 = ModifiedLogUniform(np.std(val1), 2*val1.ptp())
    weta2_1 = stats.loguniform(np.ediff1d(time).mean(), 10*time.ptp())
    weta1_2 = ModifiedLogUniform(np.std(val2), 2*val2.ptp())
    weta2_2 = stats.loguniform(np.ediff1d(time).mean(), 10*time.ptp())
    weta1_3 = ModifiedLogUniform(np.std(val3), 2*val3.ptp())
    weta2_3 = stats.loguniform(np.ediff1d(time).mean(), 10*time.ptp())
    if RVs:
        priorArray.append(weta1_1)
        priorArray.append(weta2_1)
    if BIS:
        priorArray.append(weta1_2)
        priorArray.append(weta2_2)
    if FWHM:
        priorArray.append(weta1_3)
        priorArray.append(weta2_3)

        
    ### mean function
    slope1 = stats.norm(0, val1.ptp()/time.ptp())
    offset1 = stats.uniform(val1.min(), val1.max() -val1.min())
    slope2 = stats.norm(0, val2.ptp()/time.ptp())
    offset2 = stats.uniform(val2.min(), val2.max() -val2.min())
    slope3 = stats.norm(0, val3.ptp()/time.ptp())
    offset3 = stats.uniform(val3.min(), val3.max() -val3.min())

    if RVs:
        priorArray.append(slope1)
        priorArray.append(offset1)
    if BIS:
        priorArray.append(slope2)
        priorArray.append(offset2)
    if FWHM:
        priorArray.append(slope3)
        priorArray.append(offset3)

    
    ### jitter
    jitt1 = ModifiedLogUniform(np.std(val1), 2*val1.ptp())
    jitt2 = ModifiedLogUniform(np.std(val2), 2*val2.ptp())
    jitt3 = ModifiedLogUniform(np.std(val3), 2*val3.ptp())

    if RVs:
        priorArray.append(jitt1)
    if BIS:
        priorArray.append(jitt2)
    if FWHM:
        priorArray.append(jitt3)

    return np.array(priorArray)
