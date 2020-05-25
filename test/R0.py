import numpy as np

x=[10.26,11.69,10.65,12.06,12.84,12.05,11.08,10.2,5.96,6.05,6.22,9.03,9.47,7.22,4.89,4.4,3.72,4.61,5.54,6.78,7.49,5.13,3.79,3.9,5.12,5.68,5.89,5.53,4.68,2.79,3.27,3.95,4.75,5.59,3.45,3.04,2.69,2.88,4.32,6.29,6.34,6.82,3.73,2.05,5.63,5.33,6.56,6.70,4.55,4.39,2.62,4.46,6.93,6.76,4.34,6.36,4.43]



x=np.array(x)
y=[]
for i in range(len(x)-6):
    xMean=x[i:i+7].mean()/100*14.8
    y.append(xMean)



import matplotlib.pyplot as plt


plt.plot(y)
plt.grid()
plt.title('Basic reproduction number in Germany')
plt.ylabel('Basic reproduction number')
plt.xlabel('Days form Apr. 4 to May. 20')

plt.show()
