

import pandas as pd
import matplotlib.pyplot as plt

fSig='/home/i/IGSI/data/data/csv/train/T06_T06_45_Dch/0019456_0020480.csv'
fBkg='/home/i/IGSI/data/data/csv/train/T06_DPM_45/0019456_0020480.csv'

iSig=pd.read_csv(fSig)
iBkg=pd.read_csv(fBkg)


iSig['xm'].hist(bins=100,histtype='step')
iBkg['xm'].hist(bins=100,histtype='step')


plt.show()
#
