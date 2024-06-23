
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import os
import glob
import gc


def strip_n_fill(df):
    '''This function only keeps the variables that appear to be needed.
    1. It strips extra data out of the file
    2. fills in NaNs'''
    
    
    #output value to model with NN
    Tlist = ['ALT']
    
    #list of what I think are the dependent variables to create a model for T.
    Xlist = ['time', 'RALT', 'PSA', 'PI', 'PT', 'ALTR', 'IVV', 'VSPS', 'VRTG', 'LATG', 'LONG', 
             'FPAC', 'BLAC', 'CTAC', 'TAS', 'CAS', 'GS', 'CASS', 'WS', 'PTCH', 'ROLL', 'DA', 'TAT', 
             'SAT', 'LATP', 'LONP']
    
    #later I'll do this with all the variables and see if it mattered.
    #another exploration should include the use of encoder/decoder NN to find the dependent variables.
    
    #removing the NaNs in the data.
    #Using linear interpolation to ensure there are no NaNs in my matrices used for the NN
    #thinking about stripping them out at the ETL stage for more efficiency.
    
    #linear interpolation is good for now, since most of these data rates have 0.5 secs between values.
    #it may be a good idea to use quadratic for the lower data rates and linear for the higher data rates.
    # this is another area to explore later.
    method = 'linear'
    
    #after interpolation backfill all NaN's at the beginning of the datafile. The NaNs at the end are handled
    # with the interpolation technique
    Tdf = df[Tlist].interpolate(method=method).bfill(axis=0)
    Xdf = df[Xlist].interpolate(method=method).bfill(axis=0) #saving some space
    return Xdf, Tdf


def scale_data(Xdf, Tdf, scaleX=None, scaleT=None):
    '''This function turns the dataframe into an np.array and scales it between values of 0 & 1.
    It returns the new arrays and the scale objects so it can un-scale data when needed.
    This also returns an un-scaled time to be used for plots
    1. converts df to np.array
    2. scales the values between 0 & 1
    3. If a scaleX and scaleY are given as inputs then this will use those to scale the data
       turning the dataframes into numpy arrays'''
    
    X = Xdf.iloc[:,:].values
    T = Tdf.iloc[:,:].values
    time = Xdf.iloc[:,0].values #saving for plotting later
  
    #check to see if scaleX was defined or not
    if not scaleX:
        #Normalizing/Scaling the Training data
        scaleX = MinMaxScaler()
        scaleT = MinMaxScaler()
   
        #scale the whole dataset based on it's min/max
        Xs = scaleX.fit_transform(X)
        Ts = scaleT.fit_transform(T)
        #not scaling the time variable since it's included in X and I'm going to use it later in plotting
    else:
        Xs = scaleX.transform(X)
        Ts = scaleT.transform(T)
    
    return Xs, Ts, scaleX, scaleT, time
    
#create the 3 dimensional array of the input based on the sequence_length time window
#this works with multi-dimensional X values and will assume the sequence length travels in the 
#row direction
def sliding_window(X, T, TIME, seq_length):
    x = []
    t = []
    time = []
    for i in range(len(X)-seq_length-1):
        _x = X[i:(i+seq_length),:]
        _t = T[i+seq_length,:]
        _tm= TIME[i+seq_length]
        x.append(_x)
        t.append(_t)
        time.append(_tm)

    return np.array(x),np.array(t),np.array(time)

def get_batch(X, T, batch_size=100):
    n_samples = X.shape[0]
    rows = np.arange(n_samples)
    #it shouldn't matter if this is random since the window function created a 'time history'
    np.random.shuffle(rows) 
    X = X[rows,:,:]
    T = T[rows,:]
    for first in range(0, n_samples, batch_size):
        last = first + batch_size
        yield X[first:last], T[first:last]
    # should return last batch of n_samples not evenly divided by batch_size

def read_parquet(n_files, seq_length, device, scaleX=None, scaleT=None):

    cwd = os.getcwd()
    fdir = 'Tail_687_1_parquet'
    #example parquet fdir for plotting

    
    os.chdir(os.path.join(cwd,fdir))
    fulllist = glob.glob(f'*.parquet')
    os.chdir(cwd)
    total_nfiles = len(fulllist)

  
    #randomize the starting and run through all files eventually
    ifiles = np.random.randint(0,total_nfiles,n_files)
    filelist = [fulllist[i] for i in ifiles]
   
    #creating the training set    
    #initialize these dataframes
    Xdf = pd.DataFrame()
    Tdf = pd.DataFrame()
    #print('Reading Training Files:')
    for file in filelist:
        #print(f'file:{file}')
        pname = os.path.join(cwd,fdir,file)
        df = pd.read_parquet(path=pname)

        Xdfnew, Tdfnew = strip_n_fill(df)

        #combine all training dataframes
        Xdf = Xdf.append(Xdfnew)
        Tdf = Tdf.append(Tdfnew)

    #reset the index and remove the extra column it creates.
    Xdf.reset_index(inplace=True)
    Xdf.pop('index')
    Tdf.reset_index(inplace=True)
    Tdf.pop('index')
    #print('scaling')
    
    if not scaleX:
        #scale all training data once it was combined    
        Xs, Ts, scaleX, scaleT, time = scale_data(Xdf, Tdf)
    else:
        #scale all training data once it was combined    
        Xs, Ts, scaleX, scaleT, time = scale_data(Xdf, Tdf,scaleX=scaleX,scaleT=scaleT)
        
    #print('windowing')
    #create the sliding window matrices
    Xwin, Twin, Timetrain = sliding_window(Xs,Ts,time,seq_length)
    #print('tensors')
    #training dataset
    Xtrain = torch.from_numpy(Xwin.astype(np.float32)).to(device)
    Ttrain = torch.from_numpy(Twin.astype(np.float32)).to(device)

    #print('Done with Data Loading!')
    #reclaim memory from numpy arrays
    #del Xs, Ts, time

    #reclaim memory from dataframes
    del Xdf, Tdf, df, Xdfnew, Tdfnew
    #garbage collect
    gc.collect() 
    #reset anything left to null
    Xdf = pd.DataFrame()
    Tdf = pd.DataFrame()
    df = pd.DataFrame()
    Xdfnew = pd.DataFrame()
    Tdfnew = pd.DataFrame()

    return Xtrain, Ttrain, Timetrain, scaleX, scaleT
