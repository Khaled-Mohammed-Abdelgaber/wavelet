# -*- coding: utf-8 -*-
"""
Created on Sat May 14 10:18:45 2022

@author: khali
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pywt
 

data = pd.read_excel("train.xlsx")
data = data /data.max(axis = 1)


features = []
index = []


for i in range(2*len(data.loc[1,:])):
    if i % 2 == 0:
        index.append("cA")  
        features.append(pywt.dwt(data.loc[:,int(i/2)], wavelet = "sym2")[0])
    else:
        index.append("cD")  
        features.append(pywt.dwt(data.loc[:,int(i/2)], wavelet = "sym2")[1])


train_sets = pd.DataFrame(data = features )
train_sets = train_sets.dropna()
outside = []
inside = []

for i in range(600):
    if i % 2 == 0:
        outside.append(int(i/2))
        inside.append('cA')
    else:
        outside.append(int(i/2))
        inside.append('cD')
        

hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)

i = 0

train_sets.set_index(hier_index,inplace = True)
plt.plot(10*train_sets.loc[i,:].loc['cD',:] + 0.475,color = 'r')
plt.plot(train_sets.loc[i,:].loc['cA',:]-0.2,color = 'b')
plt.plot(data.loc[:,i],color = 'k')

#===========================
x = np.zeros([1,50])
x= np.append(x , data.loc[:,1])

wavelet_trans= pywt.swt(x, wavelet = "sym2")
cAx , cDx = wavelet_trans[0][0],wavelet_trans[0][1]

plt.plot(train_sets.loc[1,:].loc['cA',:],color = 'r')
plt.plot(cAx,color = 'b')
plt.plot(data.loc[:,1],color = 'k')
plt.plot(x[:],color = 'g')

    
        
 

