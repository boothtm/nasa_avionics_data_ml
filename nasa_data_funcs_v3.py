
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
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
    
    #curious if stripping out all the data while the plane is on the ground would help.
    # this would be a good place to do it
    
    #removing the NaNs in the data.
    #Using linear interpolation to ensure there are no NaNs in my matrices used for the NN
    #thinking about stripping them out at the ETL stage for more efficiency.
    
    #linear interpolation is good for now, since most of these data rates have 0.5 secs between values.
    #it may be a good idea to use quadratic for the lower data rates and linear for the higher data rates.
    # this is another area to explore later.
    ##### Need to check the validity of linear interpolation when running in real-time that won't have access 
    #####  to the next value in order to interpolate.  should look at a plain 'pad' fill method instead.
    #method = 'linear'
    method = 'pad'
    
    #first limit the dataframe to only the columns listed above
    #after interpolation backfill all NaN's at the beginning of the datafile. The NaNs at the end are handled
    # with the interpolation technique
    Tdf = df[Tlist].interpolate(method=method).bfill(axis=0)
    Xdf = df[Xlist].interpolate(method=method).bfill(axis=0) #saving some space
    
    
    #using the Isolation Forest model to remove outliers in some of the data
    #contamination values that have been manually chosen
    vrtg_cont = float(0.03)
    latg_cont = float(0.015)
    long_cont = float(0.015)

   
    #Isolation Forest model
    model=IsolationForest(n_estimators=100, max_samples=256, contamination=vrtg_cont, max_features=1.0)
    outliers = model.fit_predict(Xdf[['VRTG']].values)
    #filtering out the outliers
    Xdf['VRTG'][outliers<0]=np.NaN
    
    #Isolation Forest model
    model=IsolationForest(n_estimators=100, max_samples=256, contamination=latg_cont, max_features=1.0)
    outliers = model.fit_predict(Xdf[['LATG']].values)
    #filtering out the outliers
    Xdf['LATG'][outliers<0]=np.NaN
    
    #Isolation Forest model
    model=IsolationForest(n_estimators=100, max_samples=256, contamination=long_cont, max_features=1.0)
    outliers = model.fit_predict(Xdf[['LONG']].values)
    #filtering out the outliers
    Xdf['LONG'][outliers<0]=np.NaN
    
    
    #pad all NaNs created by the outlier filter and backfill all NaN's at the beginning of the datafile
    Xdf = Xdf.interpolate(method=method).bfill(axis=0) #saving some space
    
    return Xdf, Tdf


def scale_data(Xdf, Tdf, scaleX=None, scaleT=None, debug=False):
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
        if debug: print('...Creating scale factors.')
        #Normalizing/Scaling the Training data
        scaleX = MinMaxScaler()
        scaleT = MinMaxScaler()
   
        #scale the whole dataset based on it's min/max
        Xs = scaleX.fit_transform(X)
        Ts = scaleT.fit_transform(T)
        #not scaling the time variable since it's included in X and I'm going to use it later in plotting
    else:
        if debug: print('scaling by initial factors...')
        Xs = scaleX.transform(X)
        Ts = scaleT.transform(T)
    
    return Xs, Ts, scaleX, scaleT, time
    
#create the 3 dimensional array of the input based on the sequence_length time window
#this works with multi-dimensional X values and will assume the sequence length travels in the 
#row direction
def sliding_window(X, T, TIME, seq_length):
    '''This function creates a 3D array that includes the sliding window for each variable at each time.
    It assumes that X and T have already been scaled appropriately.
    The unscaled TIME input is used to create a regular 1D array of real time for use in plots later.
    '''
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

def read_parquet_batch(cwd, fdir, file_list, seq_length, scaleX=None, scaleT=None, debug=False):
    '''read files from the file_list and return an array of panda dataframes so it can be 
    read into the get batch function and each df in the array can get extracted and 
    converted to a tensor that is pushed to the device during loops of the epochs
    '''
    #initialize these dataframes
    Xdf = pd.DataFrame()
    Tdf = pd.DataFrame()
    
    #initialize the dataframe arrays
    XA = []
    TA = []
    TimeA = []
    
    #read in the files and create an array of panda dataframes.
    for file in file_list:
        #print(f'file:{file}')
        pname = os.path.join(cwd,fdir,file)
        df = pd.read_parquet(path=pname)

        if debug: print('strip-n-fill')
        
        #striping out all un-needed columns and filling blank spaces with interpolated data
        Xdf, Tdf = strip_n_fill(df)

        #reset the index and remove the extra column it creates.
        Xdf.reset_index(inplace=True)
        Xdf.pop('index')
        Tdf.reset_index(inplace=True)
        Tdf.pop('index')
        
        if debug: print('Scaling')

        #if this is training data, then this will scale all flight data by the first flight file
        # it's not perfect if the first flight file sucks, but it should be close if it's a decent file.
        #if it's testing data it will be scaled by the training data scale
        if not scaleX:
            #scale all training data once it was combined    
            Xs, Ts, scaleX, scaleT, time = scale_data(Xdf, Tdf, debug=debug)
        else:
            #scale all training data once it was combined    
            Xs, Ts, scaleX, scaleT, time = scale_data(Xdf, Tdf, scaleX=scaleX, scaleT=scaleT, debug=debug)

        if debug: print('windowing')
        #create the sliding window matrices
        # Timetrain is only used for plotting purposes
        Xwin, Twin, Timetrain = sliding_window(Xs,Ts,time,seq_length)
        
        
        
        #combine all processed dataframes into an array
        XA.append(Xwin)
        TA.append(Twin)
        TimeA.append(Timetrain)
        

    #print('Done with Data Loading!')
    #reclaim memory from numpy arrays
    #del Xs, Ts, time

    #reclaim memory from dataframes
    del Xdf, Tdf, df, Xwin, Twin
    #garbage collect
    gc.collect() 
    #reset anything left to null to ensure no data is left
    Xdf = pd.DataFrame()
    Tdf = pd.DataFrame()
    df = pd.DataFrame()
    Xwin = pd.DataFrame()
    Twin = pd.DataFrame()
    

    return XA, TA, TimeA, scaleX, scaleT


def read_parquet_rand(n_files, seq_length, device, scaleX=None, scaleT=None):
    import torch
    
    #read random n_files and return the handle to the tensor that has been pushed to the 
    # device
    cwd = os.getcwd()
    fdir = 'Tail_687_1_parquet'
    #example parquet fdir for plotting

    
    os.chdir(os.path.join(cwd,fdir))
    #fulllist = glob.glob(f'*.parquet')
    os.chdir(cwd)
    

    #modifying this so I can have better one-to-one comparison of methods.
    fulllist    = ["687200107192334.parquet", 
                    "687200107301239.parquet",
                    "687200104261527.parquet",
                    "687200107251002.parquet",
                    "687200104301119.parquet",
                    "687200107101600.parquet",
                    "687200104170717.parquet",
                    "687200107181544.parquet",
                    "687200104202027.parquet",
                    "687200107170234.parquet",
                    "687200107251652.parquet",
                    "687200107122323.parquet",
                    "687200104162039.parquet",
                    "687200107311025.parquet",
                    "687200104181334.parquet",
                    "687200107171131.parquet",
                    "687200104181127.parquet",
                    "687200107241524.parquet",
                    "687200107060930.parquet",
                    "687200107150546.parquet"]
    
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
