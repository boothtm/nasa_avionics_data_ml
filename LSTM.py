
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, n_input, n_out, hidden_units, n_layers, device):
        super(LSTM, self).__init__()
        ''' #Documentation for LSTM Model:
        #This function calls Torch.nn.LSTM to train a neural network using the LSTM approach.
        
        #This object class also stores 'Model Card' information inside the model for use later:
        
        #----------------------Individual Model Properties----------------
        n_input 
        n_out 
        hidden_units 
        n_layers 
        
        seq_length 
        learning_rate 
        n_epochs 
        approach 
       
        #variables to help restart the model
        optimizer_state_dict  #pytorch optimizer object dictionary
        model_state_dict  
        loss 
        
        batch_size 
        n_train_batches 
        
        nfiles 
        train_files 
        test_files  
        
        #scaling factors used when training the model or using the model
        # Training example: 
        #   input_scaled = scaleX.fit_transform(input)
        # Usage example: 
        #   output = model(target_scaled) 
        #   target_alt = scaleT.inverse_transform(output)
        #)
        scaleX 
        scaleT 
        
        #----------------Single Model Quantitative Analyses----------------------
        test_mse 
        train_mse 
        error_trace 
        
        #error between the target test altitude and model test altitude in real units (ft)
        test_err_max 
        test_err_min 
        test_err_mean 
        test_err_std 
       
        #The test dataset target (T) altitude and the test dataset model output (Y) altitude
        Ttest_alt 
        Ytest_alt 
        
       
        #model performance characteristics
        unit_processing  #amount of time to process model per second of data
        model_size  #size of model allocated in memory on device in bytes w/o model card info
        model_size_wcard  #size of model allocated in memory on device in bytes w/model card info
        
        model_train_time   #amount of time in seconds to train this model
       
        #device properties that the model was trained on
        device 
        device_type 
        device_name 
        device_major 
        device_total_memory 
        device_multiprocessor_count 
        
        #----------------Aggregated Quantitative Analyses----------------------
        
        #number of boot loops run to gather an average MSE
        nbootloops 
        bootloop_avg_test_mse 
        bootloop_avg_train_mse 

        #testing bootstrapped data of:
        #  MSE for all previous models like this.  (list)
        self.boot_test_MSE = []
        #  max error over all previous models (array)
        self.boot_test_maxDiff = []
        #  min error over all previous models (array)
        self.boot_test_minDiff = []
        #  std dev over all previous models (array)
        self.boot_test_stdDevDiff = []
        #  testing files (2D list of strings)
        self.boot_test_files = []
         
        #training bootstrapped data of:
        #  MSE convergence data for all previous models (2D array)
        self.boot_train_MSEtrace = []
        #  final MSE for all previous models like this (array)
        self.boot_train_MSE = []
        #  max error over all previous models (array)
        self.boot_train_maxDiff = []
        #  min error over all previous models (array)
        self.boot_train_minDiff = []
        #  std dev over all previous models (array)
        self.boot_train_stdDevDiff = []
        #  testing files (2D list of strings)
        self.boot_train_files = []
        #  nEpochs used (array of ints)
        self.nEpochs = 0
    
        
        ''' 
       
        #----------------------Individual Model Properties----------------
        self.n_input = n_input
        self.n_out = n_out
        self.hidden_units = hidden_units
        self.n_layers = n_layers
        
        self.seq_length = None
        self.learning_rate = None
        self.n_epochs = None
        self.approach = None
       
        #variables to help restart the model
        self.optimizer_state_dict = None  #pytorch optimizer object dictionary
        self.model_state_dict  = None
        self.loss = None
        
        self.batch_size = None
        self.n_train_batches = None
        
        self.nfiles = None
        self.train_files = []
        self.test_files  = []
        
        self.scaleX = None
        self.scaleT = None
        
        #---------------Training Device Properties----------------------------
        #device properties
        self.device = device
        self.device_type = device.type
        self.device_name = None
        self.device_major = None
        self.device_total_memory = None
        self.device_multiprocessor_count = None
        
        #----------------Single Model Quantitative Analyses----------------------
        self.test_mse = None
        self.train_mse = None
        
        #error between the target test altitude and model test altitude in real units (ft)
        self.test_err_max = None
        self.test_err_min = None
        self.test_err_mean = None
        self.test_err_std = None
        
        self.Ttest_alt = None
        self.Ytest_alt = None
        
        self.unit_processing = None #amount of time to process model per second of data
        self.model_size = None #size of model allocated in memory on device in bytes w/o model card info
        self.model_size_wcard = None #size of model allocated in memory on device in bytes w/model card info
        
        self.model_train_time = None  #amount of time in seconds to train this model
       
        
        self.error_trace = []
        
        #----------------Aggregated Quantitative Analyses----------------------
  
        #these variables are for the models in the bootstrap aggegation (bagging)

        #number of boot loops run
        self.nbootloops = None
        self.bootloop_avg_train_mse = None
        
      
        #testing bagging data of:
        # mean mse over all models in the bootloop
        self.bootloop_avg_test_mse = None
        #  MSE for all previous models like this.  (list)
        self.bootloop_test_mse_list = []
        
        #error between the target test altitude and model test altitude in real units (ft)
        # for all other models in the bagging
        #  max error over all previous models (array)
        self.bootloop_test_err_max_list = []
        #  min error over all previous models (array)
        self.bootloop_test_err_min_list = []
        #  std dev over all previous models (array)
        self.bootloop_test_err_std_list = []
        #  mean over all previous models (array)
        self.bootloop_test_err_mean_list = []
        
        #  testing files (2D list of strings)
        self.boot_test_file_lists = []
         
        #training bagging data of:
        #  MSE convergence data for all previous models (2D array)
        self.bootloop_train_error_trace_list = []
        #  final MSE for all models in the bagging(array)
        self.bootloop_train_mse_list = []
        #  max error over all previous models (array)
        self.bootloop_train_err_max_list = []
        #  min error over all previous models (array)
        self.bootloop_train_err_min_list = []
        #  std dev over all previous models (array)
        self.bootloop_train_err_std_list = []
        #  mean over all previous models (array)
        self.bootloop_train_err_mean_list = []
        
        #  testing files (2D list of strings)
        self.bootloop_train_file_lists = []
        
        #  nEpochs used for each(array of ints)
        self.bootloop_epochs_list = []
    
        #Long Short Term Memory is supposed to be better than the original multi-layer Elman RNN
        # with tanh or ReLU non-linearity to an input sequence.
        self.lstm = nn.LSTM(input_size=n_input, hidden_size=hidden_units,
                            num_layers=n_layers, batch_first=True)
    
        #this what the forward function will call
        # itapplies a linear tranformation to the incoming layers (other option would be BiLinear)
        # includes the bias weights
        self.fc = nn.Linear(hidden_units, n_out)
    
    def __repr__(self):
        return f'{type(self).__name__}({self.approach}, {self.n_input}, ' + \
            f'{self.n_out}, {self.seq_length}, {self.learning_rate}, {self.hidden_units}, {self.n_layers} )'

    def __str__(self):
        s = self.__repr__()
        if self.total_epochs > 0:
            s += f'\n Trained for {self.total_epochs} epochs.'
            s += f'\n Final objective value is {self.error_trace[-1]:.4g}.'
        return s
    
    
    def forward(self, x):
        #nn.LSTM defaults to zeros for h_0 and c_0 if not provided, so i'm not sure why these are here.
        h_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_units).to(self.device)
        c_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_units).to(self.device)
        
        ## Propagate input through LSTM
        #ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        #h_out = h_out.view(-1, self.hidden_units)
        
        # Propagate input through LSTM
        out, hidden = self.lstm(x, (h_0, c_0))
        
        #call the linear transformation on the output of the hidden layers
        out = self.fc(out[:,-1,:])
        
        return out
