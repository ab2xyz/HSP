import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('/home/i/IGSI/data/data/csv/train/T02_T02_45_pp_Etac/0001024_0002048.csv')['xm']

data.hist(bins=100)
plt.title('xm @ pp_Etac')

plt.show()
