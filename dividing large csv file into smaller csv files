# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 22:09:37 2022

@author: Khaled
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 


ppgDf = pd.read_csv("cleaned data/Record_index.csv",chunksize=10000,encoding="utf-8")
for i , f in enumerate(ppgDf):
    f.to_csv(rf"cleaned data/{i}.csv")
    
    
