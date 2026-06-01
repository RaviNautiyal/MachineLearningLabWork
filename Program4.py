import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
iq = np.array([1.5,2.4,3.7,4.7 , 7.85,6.5,7.2,8.6,9.5,10.0])
placement =np.array([2, 4, 3.4, 7,6,8 , 9, 10, 11,12])
plt.scatter(iq, placement)
meanx=np.mean(iq)
print(meanx)
meany = np.mean(placement)
print(meany)
Function for finding the m and b
rows = iq.shape[0]
print(rows)
m= 0
b = 0

num= 0
den = 0
for i in range(rows):
  num += (placement[i]-meany)  * (iq[i]- meanx)
  den += (iq[i]- meanx)**2
m = num/den
b = meany - (m*meanx)
print(m,b)

print(m)
print(b)
m1= 0.5
b1 = 1
ypredicted1 = m1*xactual + b1
print(m1,b1)

xactual = np.array([2, 4, 5, 7, 8, 9])
yactual = np.array([3, 7 , 8, 9,9,11])
plt.scatter(xactual , yactual)
ypredicted = m*xactual + b
plt.plot(xactual, ypredicted,color = "red")
plt.plot(xactual , ypredicted1, color = "green")

dataset = pd.read_csv("placements.csv")