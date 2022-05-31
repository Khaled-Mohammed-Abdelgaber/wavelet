# -*- coding: utf-8 -*-
"""
Created on Thu May 19 22:33:26 2022

@author: Khaled
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from kymatio.numpy  import Scattering1D as S1D
import function as fn 


data = pd.read_excel("detrended_normalized.xlsx")  # preprocessed data

X = data.loc[:,101]   # one sample  to investigate wavelet scattering on it

plt.plot(data.loc[:,1000]) # plot the signal 

   

X = X.to_numpy()
X = np.transpose(X)
X = np.delete(X ,range(278,300))
X = np.delete(X,range(0,22))
plt.plot(X)    


T =2**8        # signal length must be 2**() i made signal length to 256

J = 3          # maximum scale is 2**j must be less than T
Q = 4           # by observation we found that almost smallest feature are in 9 samples
                #

 
plt.plot(X) # plot the signal 

S = S1D(J, T, Q)

meta = S.meta()
order0 = np.where(meta['order'] == 0)
order1 = np.where(meta['order'] == 1)
order2 = np.where(meta['order'] == 2)

Sdata = S.scattering(X)

plt.figure(figsize=(8, 8))
plt.subplot(4, 1, 1)
plt.plot(X)
plt.title('signal')

plt.subplot(4, 1, 2)

plt.plot(Sdata[order0][0])
plt.title('Zeroth-order scattering')
plt.subplot(4, 1, 3)
plt.imshow(Sdata[order1], aspect='auto')
plt.title('First-order scattering')
plt.subplot(4, 1, 4)
plt.imshow(Sdata[order2], aspect='auto')
plt.title('Second-order scattering')

plt.show()





