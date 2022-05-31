# -*- coding: utf-8 -*-
"""
Created on Sun May 29 21:15:33 2022

@author: Khaled

"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import statsmodels.api as sm 
from obspy.signal.detrend import polynomial
from kymatio.numpy  import Scattering1D as S1D

#============================================================================

#function to divide datasets into x and y (features and output)

def dataset_dis(csv_data_path):
    
    data = pd.read_csv(csv_data_path)
    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values
    y = y.reshape(len(y),1)
    
    return X , y

#============================================================================

#function to onehot encode categorical data
def dummyEncoding( X , column_number ,rem = "passthrough" ):
    CT = ColumnTransformer(transformers = [("encoder",OneHotEncoder(),[column_number])],remainder = rem) 
    X = np.array(CT.fit_transform(X))
    return X

#============================================================================

def missing_value_computation(X,range_ ,missing_values_type = np.nan,Strategy = "mean"):
    initial = range_[0]
    final = range_[1]
    imputer = SimpleImputer(missing_values = missing_values_type ,strategy = Strategy )
    imputer.fit(X[:,initial:final])
    X[:,initial:final] = imputer.transform(X[:,initial:final])
    
    return X

#=========================================================================

    
def auto_OLS_model(X,y,SL):
    
    X = np.array(X, dtype=float)
    regressor_OLS = sm.OLS( y,X).fit()
    print(regressor_OLS.summary()) 
    k = X.shape[1]       # k = 6
    if(max(regressor_OLS.pvalues) > SL ):
        for i in range(X.shape[1]):
            k = k - 1
            for j in list(range(k)):
                if(regressor_OLS.pvalues[j] >= max(regressor_OLS.pvalues)):
                    X = np.delete(X,j,axis = 1)
                    X = np.array(X, dtype=float)
                    regressor_OLS = sm.OLS( y,X).fit()
                    print(regressor_OLS.summary())
                    break
    print(regressor_OLS.summary())                
    return X
#==========================================================================
#to write data to excel file 
def excel_write(file_name , X ):
    if(type(X) == pd.core.frame.DataFrame ):
        X.to_excel(file_name+".xlsx")
    else:
        print("deal only with pandas data frame")

#============================================================
#to normalize data values
def normalizer(X):
    if(type(X) == pd.core.frame.DataFrame ):
        for i in range(len(X.columns)):
            X.loc[:,i] = X.loc[:,i] /X.loc[:,i].abs().max()
            
    return X 
#==============================================================
def detrend_and_normalize(X):
    if(type(X) == pd.core.frame.DataFrame ):
        for i in range(1,len(X.columns)):
            X.loc[:,i] = polynomial(X.loc[:,i], order=3, plot=False)  
            X.loc[:,i] = X.loc[:,i] /X.loc[:,i].abs().max()
    else:
             print("deal only with pandas data frame")
             
    return X             

#===================================================================
"""
function to convert pandas Series to numpy array make transposation if needed and also padding 
or shortening if needed 

if trans = true this mean series need to transpose this case occur when 
datasets stored as columns in pandasDataFrame

length represent the required length  to be fed to wavelet scatter


"""


def convert_to_numpy(S ,length, trans= False):
    S = S.to_numpy()
    if(trans == True):
        S = np.transpose(S)
    if(length < len(S)):
        diff = len(S) - length
        S = np.delete(S ,range(len(S) - int(diff / 2) ,len(S)))
        S = np.delete(S,range(0,int(diff / 2)))
    elif(length > len(S)):
        diff = length - len(S) 
        S = np.pad(S, (int(diff / 2),int(diff / 2)), mode='constant')
    return S

#======================================================================

def seriersCoeff(S , ax , j ,q ,t , order ):
    if(ax == 0):
        X = convert_to_numpy(S,t,trans = False)
    else :
        X = convert_to_numpy(S,t,trans = True)
    scatter = S1D(j, t, q)
    meta = scatter.meta()
    order0 = np.where(meta['order'] == 0)
    order1 = np.where(meta['order'] == 1)
    order2 = np.where(meta['order'] == 2)
    Sdata = scatter.scattering(X)
    if(order == 0):
        return Sdata[order0][0]
    elif(order == 1):
        return Sdata[order1]
    else:
        return Sdata[order2]
        
#=====================================================================
"""
the function will return a list that contain all coeeffient of a specific order that 
resulted from wavelet transsform 

df ====> represent the data from excel 
datasetNum ====> represent the number of datasets
T======> length of each signal 
J ======> maximum scale is 2**j must be less than T
axis =====> if axis = 0 mean one dataset stored as row 
            if axis = 1 mean one dataset stored as column
Q =====> * controls the number of wavelets per octave in the first-order filter bank.
        * The larger the value, the narrower these filters are in the frequency domain 
          and the wider they are in the time domain
        * the number of non-negligible oscillations in time is proportional to Q
        * For audio signals, it is often beneficial to have a large value for Q 
        (between 4 and 16), since these signals are often highly oscillatory
        and are better localized in frequency than they are in time.
        Note that it is currently not possible to control the number
        of wavelets per octave in the second-order filter bank, which is fixed to one.
order ===> represent the needed order to be returned

"""
def DfCoeff(df,datasetnum,axis , J , Q , T,Order):
    df = pd.DataFrame(data=df.values,columns=   range(datasetnum))
    CoeffiecentList = []
    if(axis == 0):
        for i in range(datasetnum):
           CoeffiecentList.append(seriersCoeff(df.loc[i,:],ax = axis ,j=J ,q=Q,t=T ,order = Order )) 
    else:
        for k in range(datasetnum):
           CoeffiecentList.append(seriersCoeff(df.loc[:,k],ax = axis ,j=J ,q=Q,t=T ,order = Order )) 
    return CoeffiecentList
  
    
    
    


















