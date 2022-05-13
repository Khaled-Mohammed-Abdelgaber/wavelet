# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:26:59 2022

@author: khali
"""

import pywt
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

samplerat , data = wavfile.read("file.wav")
t = np.arange(len(data))/float(samplerat)
data = data / data.max()
cA , cD = pywt.dwt(data, "bior6.8","per")
y = pywt.idwt(cA, cD, "bior6.8","per")
wavfile.write("fileRec.wav", samplerat, y)
wavfile.write("fileCD.wav", samplerat, cD)

#plotting

l = len(data)
y = y[0:l]
plt.figure(figsize = (30,20))
plt.subplot(4,1,1)
plt.plot(t,data,color = "k")
plt.xlabel("time")
plt.ylabel("S")
plt.title("Original Signal")


plt.subplot(4,1,2)
plt.plot( cA,color = "r")
plt.xlabel("Samples")
plt.ylabel("cA")
plt.title("Approximation Coef: cA")


plt.subplot(4,1,3)
plt.plot(cD,color = "g")
plt.xlabel("samples")
plt.ylabel("cD")
plt.title("Detailed Coef: cD")



plt.subplot(4,1,4)
plt.plot(t,y,color = "b")
plt.xlabel("time")
plt.ylabel("signal")
plt.title("Reconstructed signal")

plt.savefig("plot.png", dpi = 100)
plt.show()






