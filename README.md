# wavelet
the purpose of this respo. is to save wavelet codes file  
## wavelet.py
this file contain a code that use single wavelet transform of audio signal .wav file and reconstruction and ploting coefficients 
## detrended_normalized 
this a dertrended and normalized version of train data using obspy.signal.detrend.polynomial
## functions.py
contain functions such as : 
### convert_to_numpy:
function to convert pandas Series to numpy array make transposation if needed and also padding 
or shortening if needed 
if trans = true this mean series need to transpose this case occur when 
datasets stored as columns in pandasDataFrame
length represent the required length  to be fed to wavelet scatter
### DfCoeff:
the function will return a list that contain all coeeffient of a specific order that 
resulted from wavelet transsform 

#### df:
represent the data from excel 
#### datasetNum
represent the number of datasets
#### T
length of each signal 
#### J 
maximum scale is 2**j must be less than T
#### axis
if axis = 0 mean one dataset stored as row 
if axis = 1 mean one dataset stored as column
#### Q
* controls the number of wavelets per octave in the first-order filter bank.
* The larger the value, the narrower these filters are in the frequency domain 
          and the wider they are in the time domain
* the number of non-negligible oscillations in time is proportional to Q
* For audio signals, it is often beneficial to have a large value for Q 
        (between 4 and 16), since these signals are often highly oscillatory
        and are better localized in frequency than they are in time.
* Note that it is currently not possible to control the number
        of wavelets per octave in the second-order filter bank, which is fixed to one.
#### order 
represent the needed order to be returned
