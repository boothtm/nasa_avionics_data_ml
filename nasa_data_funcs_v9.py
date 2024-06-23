
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import os
import glob
import gc
import torch

def strip_n_fill(df,outliers=False,add_elev=False,VRTG=True,airborne_only=False, include_alt=False, delta=False):
    '''This function only keeps the variables that appear to be needed.
    1. It strips extra data out of the file
    2. fills in NaNs
    3. if filter is set to True then it performs an Isolation Forest filter
    4. if add_elev is true then call the function to add elevation for each lat/lon
    5. if VRTG = False then this will exclude VRTG, LATG, and LONG in the input.
    6. if include_alt = True, then ALT will be included in the inputs (XLIST)
    '''
    
    
    #output value to model with NN
    Tlist = ['ALT']
    
    
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
    
    #remove data when it's on the ground before padding any other data.
    # this is done since the NSQT switch has a periodic 0 signal when airborne and I don't want to pad
    # this noise everywhere at 16hz then strip it.  This should also happen before the elevation data is 
    # added.
    if airborne_only:
        df = strip_ground_data(df)

  
    #list of what I think are the dependent variables to create a model for T.
    if include_alt:
        if VRTG:
            Xlist = ['time', 'ALT', 'RALT', 'PSA', 'PI', 'PT', 'ALTR', 'IVV', 'VSPS', 'VRTG', 'LATG', 'LONG', 
                     'FPAC', 'BLAC', 'CTAC', 'TAS', 'CAS', 'GS', 'CASS', 'WS', 'PTCH', 'ROLL', 'DA', 'TAT', 
                     'SAT', 'LATP', 'LONP']
        else:
            #Removed the LONG, LATG, VRTG acceleration measurements that had period noise in it
            Xlist = ['time', 'ALT', 'RALT', 'PSA', 'PI', 'PT', 'ALTR', 'IVV', 'VSPS',  
                     'FPAC', 'BLAC', 'CTAC', 'TAS', 'CAS', 'GS', 'CASS', 'WS', 
                     'PTCH', 'ROLL', 'DA', 'TAT', 'SAT', 'LATP', 'LONP']
    else:
        if VRTG:
            Xlist = ['time', 'RALT', 'PSA', 'PI', 'PT', 'ALTR', 'IVV', 'VSPS', 'VRTG', 'LATG', 'LONG', 
                     'FPAC', 'BLAC', 'CTAC', 'TAS', 'CAS', 'GS', 'CASS', 'WS', 'PTCH', 'ROLL', 'DA', 'TAT', 
                     'SAT', 'LATP', 'LONP']
        else:
            #Removed the LONG, LATG, VRTG acceleration measurements that had period noise in it
            Xlist = ['time', 'RALT', 'PSA', 'PI', 'PT', 'ALTR', 'IVV', 'VSPS',  
                     'FPAC', 'BLAC', 'CTAC', 'TAS', 'CAS', 'GS', 'CASS', 'WS', 
                     'PTCH', 'ROLL', 'DA', 'TAT', 'SAT', 'LATP', 'LONP']
        
    Tdf = df[Tlist].interpolate(method=method).bfill(axis=0)
    Xdf = df[Xlist].interpolate(method=method).bfill(axis=0) #saving some space
    
    #if you would like to keep the VRTG, LATG, and LONG then filter them with Isolation Forest method 
    # ensure outliers=True
    if outliers and VRTG:
        
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
        Xdf = Xdf.interpolate(method=method).bfill(axis=0) 
        
    
    #reset the index and remove the extra column it creates.
    Xdf.reset_index(inplace=True)
    Xdf.pop('index')
    Tdf.reset_index(inplace=True)
    Tdf.pop('index')  
   
    if add_elev:
        Xdf, Tdf = add_elevation(Xdf, Tdf)
        
    
    
    return Xdf, Tdf

def strip_ground_data(df,debug=False):
    '''
    This function strips the ground data out of the panda dataframe (df)
    based on the NSQT logical data which is the "squat switch nose main gear"
    value of 0 means on the ground, value of 1 is airborne
    To be positive it is truly on the ground, NSQT is combined with the 
    groundspeed (GS). When both gs=0 or nsqt=0 it is on the ground
    
    '''
    n_samples = df.shape[0]
    if debug: print(f'n_samples = {n_samples}')
   
    ##loop through the df 
    #i=0
    #while i < (n_samples-8):
    #    if debug: print(i)
    #    nsqt = df['NSQT'][i]
    #    groundspeed = df['GS'][i]
    #    airborne = nsqt + groundspeed
    #    
    #    nsqt_p1 = df['NSQT'][i+4]
    #    groundspeed_p1 = df['GS'][i+4]
    #    airborne_p1 = nsqt_p1 + groundspeed_p1 
    #    
    #    nsqt_p2 = df['NSQT'][i+8]
    #    groundspeed_p2 = df['GS'][i+8]
    #    airborne_p2 = nsqt_p2 + groundspeed_p2 
    #    #strip if there are 3 consequtive values not airborne 
    #    #the airborne data is at 4Hz but samples are at 16Hz
    #    if (not nsqt    or not groundspeed) and \
    #       (not nsqt_p1 or not groundspeed_p1) and \
    #       (not nsqt_p2 or not groundspeed_p2):
    #        #we must remove the full row of data and next 4 rows to account for 16Hz data
    #        # at the i index in the df dataframe
    #        if debug: print(f'deleting rows {i, i+1, i+2, i+3}')
    #        df.drop([i,i+1,i+2,i+3],inplace=True)
    #        i+=4
    #    else:
    #        i+=1
    ##always remove the last 8 datapoints (at 16Hz) to catch the last 2 points (at 4 Hz) on the ground
    ##d = np.linspace(n_samples-9,n_samples-1,num=8).astype(int)
    ##df.drop(d,inplace=True)
    #df.drop([n_samples-1,n_samples-2,n_samples-3,n_samples-4],inplace=True)
    #df.drop([n_samples-5,n_samples-6,n_samples-7,n_samples-8],inplace=True)
    
    
    #just delete if ground speed is 0.
    drop_list = []
    i=0
    while i < (n_samples-4):
        if debug: print(i)
        groundspeed = df['GS'][i]
        
        #the groundspeed data is at 4Hz but samples are at 16Hz
        if not groundspeed:
            #we must remove the full row of data and next 4 rows to account for 16Hz data
            # at the i index in the df dataframe since ground speed is at 4Hz
            if debug: print(f'deleting rows {i, i+1, i+2, i+3}')
            #df.drop([i,i+1,i+2,i+3],inplace=True)  #df.drop is extremely slow
            drop_list.append(i)
            drop_list.append(i+1)
            drop_list.append(i+2)
            drop_list.append(i+3)
            i+=4
        else:
            i+=1
    
    #always clean-up the last datapoints if not already covered in the loop above
    while i < n_samples:
        #df.drop(i,inplace=True) #df.drop was slow here
        drop_list.append(i)
        i+=1
    
    #drop all the rows at once which is faster than dropping them in the while loop.
    df = df.iloc[df.index.drop(drop_list)]
    
    #if there were rows/index's that were removed, then reset the index and remove the extra column it creates.
    df.reset_index(inplace=True)
    df.pop('index')
    
    return df

def add_elevation(Xdf, Tdf, file=None, debug=False):
    '''
    This function is called by the strip_n_fill function.
    This function reads in the file and extracts the elevation of the ground
    for the given Lat/Lon in the Xdf data frame. 
    It has to use some system functions since there isn't a current python module
    to ulitize.  This means there is more I/O while reading/writing to files.
    it is assumed that the Xdf dataframe has already had all the NaNs fixed and
    contains the 'LATP' and 'LONP' data columns.
    Thinking about adding this to the ETL python file so I don't have to do this
    everytime I read in a file, but this also makes it more flexible if I want 
    to have a higher resolution elevation datafile to use.
    '''
    
    #this is a merged file containing most of the US.
    #yes i realize this is a bad habit, but I'm in a hurry unfortunately.  I'll
    # refactor my code later (haha, jokes on you future Tom!)
    if file:
        gmted_file = file
    else:
        gmted_file = '/s/chopin/b/grad/boothtm/research/nasa/GMTED2010_30n120-90merge_mea075.tif'   #high resolution
        
        
    #appending random integer to file name to prevent collisions
    rand = np.random.randint(1,5000)
    coord_file = '/tmp/lat-lon'
    coord_file = f'{coord_file}{rand}.csv'
    elevation_file = '/tmp/elev'
    elevation_file = f'{elevation_file}{rand}.csv'
    
    if debug: print(coord_file)
    
    n_samples = Xdf.shape[0]

    with open(coord_file, 'w') as f:
        for i in range(n_samples):
           
            lon = Xdf['LONP'][i]
            lat = Xdf['LATP'][i]
            
            #ensure the lat/lon is within the bounds of the gmted files or drop the data in the dataframe
            #this is hard coded to the GMTED2010_30n120-90merge_mea075.tif file
            #box bounds 
            lb = -120.0001389 #left longitude bound
            rb =  -60.0001389 #right longitude bound
            ub =   49.9998611 #upper latitude bound
            lwb=   29.9998611 #lower latitude bound 
            
            if (lat <= ub) and (lat >= lwb) and (lon <= rb) and (lon >= lb):
                f.write(f"{lon}  {lat}")
                f.write('\n')
            else:
                #we must remove the full row of data at the i index in the Xdf & Tdf dataframes
                Xdf.drop([i],inplace=True)
                Tdf.drop([i],inplace=True)
               

    #if there were rows/index's that were removed, then reset the index and remove the extra column it creates.
    Xdf.reset_index(inplace=True)
    Xdf.pop('index')
    Tdf.reset_index(inplace=True)
    Tdf.pop('index') 
    
    #running a system command since there is no direct python module, however
    # you must still 'conda install gdal -c conda-forge' to install this package 
    sysreturn = os.system(r'gdallocationinfo -valonly -wgs84 "%s" <%s >%s' % (gmted_file, coord_file, elevation_file));

    eldf = pd.read_csv(elevation_file,header=None,names=['ELEVATION'])

    #Delete the temp files
    try:
        os.remove(coord_file)
        os.remove(elevation_file)
    except OSError as e:  ## if failed, report it back to the user ##
        print ("Error: %s - %s." % (e.filename, e.strerror))

    #convert meters to feet
    eldf = eldf[:]/0.3048
    
   
    if debug:
        x_samples = Xdf.shape[0]
        e_samples = eldf.shape[0]
        print(f'Xdf samples:  {x_samples}')
        print(f'eldf samples: {e_samples}')

    #add the elevation as another column on the dataframe
    Xdf['ELEVft'] = eldf
    
    return Xdf, Tdf


def scale_data(Xdf, Tdf=pd.DataFrame(), scaleX=None, scaleT=None, debug=False):
    '''This function turns the dataframe into an np.array and scales it between values of 0 & 1.
    It returns the new arrays and the scale objects so it can un-scale data when needed.
    This also returns an un-scaled time to be used for plots
    1. converts df to np.array
    2. scales the values between 0 & 1
    3. If a scaleX and scaleY are given as inputs then this will use those to scale the data
       turning the dataframes into numpy arrays
    Note: if Tdf is not given you can just use this function to scale your inputs when using
     the model.
    '''
   
    #converting to np array
    X = Xdf.iloc[:,:].values
    T = Tdf.iloc[:,:].values
        
    
    #this is specific only to the nasa dataset
    time = Xdf.iloc[:,0].values #saving for plotting later
  
    #check to see if scaleX was defined or not
    if not scaleX:
        if debug: print('...Creating scale factors.')
        #Normalizing/Scaling the Training data
        scaleX = MinMaxScaler()
        #scale the whole dataset based on it's min/max
        Xs = scaleX.fit_transform(X)
        if Tdf.empty: 
            scaleT = None
            Ts = np.array(0)
        else:
            scaleT = MinMaxScaler()
            Ts = scaleT.fit_transform(T)
   
        #not scaling the time variable since it's included in X and I'm going to use it later in plotting
    else:
        if debug: print('scaling by initial factors...')
        Xs = scaleX.transform(X)
        if Tdf.empty: 
            Ts = np.array(0)
        else:
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

def read_parquet(cwd, fdir, file_list, seq_length=None, scaleX=None, scaleT=None, debug=False,outliers=False,\
                       add_elev=False, VRTG=False, include_alt=False, delta=False, nntype='regression', targetAltTime=0,\
                       airborne_only=False):
    '''read files from the file_list and return an array of panda dataframes so it can be 
    read into the get batch function and each df in the array can get extracted and 
    converted to a tensor that is pushed to the device during loops of the epochs
    set outliers=True to run the forest isolation
    nntype should equal on of the following: regression, recurrent
    if targetAltTime = 1, then target time is equal to t+1
    nntype is either 'regression' or 'recurrent'. regression will return a dataframe and recurrent will return an array
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
        Xdf, Tdf = strip_n_fill(df,outliers=outliers,add_elev=add_elev,VRTG=VRTG,include_alt=include_alt,delta=delta,\
                               airborne_only=airborne_only)
        
       
        #only do this windowing scheme for recurrent NN for now
        if nntype == 'recurrent':
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
                
            #create the sliding window matrices
            # Timetrain is only used for plotting purposes
            if debug: print('windowing')
            Xwin, Twin, Timetrain = sliding_window(Xs,Ts,time,seq_length)
            #combine all processed dataframes into an array
            XA.append(Xwin)
            TA.append(Twin)
            TimeA.append(Timetrain)
            del Xwin, Twin
            Xwin = pd.DataFrame()
            Twin = pd.DataFrame()
            
        elif nntype == 'regression':
           
            # the regression NN used for this data has the scaling factor built in already
            #change the target time (t)
            # if targetAltTime = 1, then target time is equal to t+1 by shifting the input
            # dataframe (Xdf) forward 1 timestep
            if not targetAltTime:
                XA = Xdf
            else:
                Xdf = Xdf.shift(periods=targetAltTime)
                #drop the first targetAltTime rows 
                drop_list = np.arange(targetAltTime)
                Xdf = Xdf.iloc[Xdf.index.drop(drop_list)]
                Tdf = Tdf.iloc[Tdf.index.drop(drop_list)]
    
                #if there were rows/index's that were removed, then reset the index and remove the extra column it creates.
                Tdf.reset_index(inplace=True)
                Xdf.reset_index(inplace=True)
                Tdf.pop('index')
                Xdf.pop('index')
                XA = Xdf
            
            TA = Tdf
            scaleX = None
            scaleY = None
            TimeA = Xdf['time']
        
        
        else:
            raise Exception("method must be 'recurrent' or 'regression'")
        
        
        
        

    #print('Done with Data Loading!')
    #reclaim memory from numpy arrays
    #del Xs, Ts, time

    #reclaim memory from dataframes
    del Xdf, Tdf, df
    #garbage collect
    gc.collect() 
    #reset anything left to null to ensure no data is left
    Xdf = pd.DataFrame()
    Tdf = pd.DataFrame()
    df = pd.DataFrame()
    

    return XA, TA, TimeA, scaleX, scaleT


def read_parquet_batch(cwd, fdir, file_list, seq_length, scaleX=None, scaleT=None, debug=False,outliers=False,\
                       add_elev=False, VRTG=False):
    '''read files from the file_list and return an array of panda dataframes so it can be 
    read into the get batch function and each df in the array can get extracted and 
    converted to a tensor that is pushed to the device during loops of the epochs
    set outliers=True to run the 
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
        Xdf, Tdf = strip_n_fill(df,outliers=outliers,add_elev=add_elev,VRTG=VRTG)
        
        #adding the elevation data to the dataframe


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

def read_parquet_flight_merge(cwd, fdir, file_list, seq_length, scaleX=None, scaleT=None, \
                              outliers=False, add_elev=False, VRTG=False, airborne_only=False):
    '''read files from the file_list and a fully merged panda dataframes so it can be 
    read into the get batch function and have small batches extracted and 
    converted to a tensor that is pushed to the device during loops of the epochs
    set outliers=True to run the Isolation Filter on the full dataset
    '''
    
    #creating the training set    
    #initialize these dataframes
    Xdf = pd.DataFrame()
    Tdf = pd.DataFrame()
    #print('Reading Training Files:')
    for file in file_list:
        #print(f'file:{file}')
        pname = os.path.join(cwd,fdir,file)
        df = pd.read_parquet(path=pname)

        Xdfnew, Tdfnew = strip_n_fill(df, outliers=outliers, add_elev=add_elev, VRTG=VRTG, \
                                      airborne_only=airborne_only)

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
    

    #print('Done with Data Loading!')
    #reclaim memory from numpy arrays
    #del Xs, Ts, time

    # I may not even need to delete these dataframes since I've moved these funcs
    # to a stand-alone python file.  I believe variables are released after it's called.
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

    return Xwin, Twin, Timetrain, scaleX, scaleT

def read_parquet_rand(n_files, seq_length, device, scaleX=None, scaleT=None):
    
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


def get_batch(X, T, batch_size=100):
    #modified function from CS545 ML
    n_samples = X.shape[0]
    #loop through all samples of dataframes by batch_size steps
    for first in range(0, n_samples, batch_size):
        last = first + batch_size
        #yield picks back where it left off when get_batch is called by the for loop
        yield X[first:last], T[first:last] 
    # should return last batch of n_samples not evenly divided by batch_size
    
    
