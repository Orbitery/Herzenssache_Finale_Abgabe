# -*- coding: utf-8 -*-

import csv
import os
from statistics import mode
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow.keras as keras
from ecgdetectors import Detectors
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
import pickle
from PCA_ import *
from save_models import *
from plotter import *
from Models_ import *
from preprocessing_ import *
import argparse

from sklearn.decomposition import PCA
from wettbewerb import load_references

### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig
parser = argparse.ArgumentParser(description='Train given Model')
parser.add_argument('--modelname', action='store',type=str,default='Resnet')
parser.add_argument('--bin', action='store',type=str,default='True')
parser.add_argument('--pca_active', action='store',type=str,default="False")
parser.add_argument('--epochs', action='store',type=int,default="10")
parser.add_argument('--batchsize', action='store',type=int,default="512")
parser.add_argument('--treesize', action='store',type=int,default=50)


args = parser.parse_args()
pca_active = args.pca_active
bin = args.bin
modelname = args.modelname
epochs_ = args.epochs
batchsize = args.batchsize
treesize_ = args.treesize

ecg_leads,ecg_labels,fs,ecg_names = load_references("../training") # Import ECG files, associated diagnostics, sampling frequency (Hz) and name.                                              # Sampling-Frequenz 300 Hz




train_labels,train_samples,r_peaks_list,X_train, y_train, X_test, y_test = Preprocessing(ecg_leads,ecg_labels,fs,ecg_names,modelname,bin)

#Check if PCA should be performed to reduce Features
if (pca_active=="True"):
    X_train, X_test = PCA_function(X_train,X_test,modelname)


start = time.perf_counter()

#Check which model is choosen
if (modelname=="CNN"):
    model, history = CNN_Model(X_train, y_train, X_test, y_test, epochs_, batchsize)
elif(modelname=="LSTM"):
    model, history = LSTM(X_train, y_train, X_test, y_test, epochs_, batchsize)
elif (modelname=="RandomForrest"):
    model, history = RandomForrest(X_train, y_train, X_test, y_test, epochs_, batchsize, treesize_)
elif (modelname=="Resnet"):
    model, history = Resnet(X_train, y_train, X_test, y_test, epochs_, batchsize)
elif (modelname=="XGboost"):
    model, history = XGBoost(X_train, y_train, X_test, y_test, epochs_, batchsize)




Saver(model, bin, modelname) #Save trained Model for later predictions

plot_creater(history,bin, modelname) #Function to plot Acc / Loss over trainingssteps

end = time.perf_counter()
perf = end-start
print(f"Time in min: {perf/60:0.2f}")
