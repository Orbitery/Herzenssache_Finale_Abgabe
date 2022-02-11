import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
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

from sklearn.decomposition import PCA


def BP_Filter(ecg_lead): 
    """[A bandpass is applied. 
    For this purpose, particularly high frequencies and particularly low frequencies are cut.
     This is to generate a resilience against errors in the data and an increased generalizability.]

    Args:
        ecg_lead ([list of numpy-Arrays]): [ECG signal]

    Returns:
        ecg_lead [list of numpy-Arrays]: [ECG signal]
    """


    fs = 300  # Sampling frequency

    fc = 30  # Cut-off frequency of the filter
    w = fc / (fs / 2) # Normalize the frequency

    b, a = signal.butter(5, w, 'low')
    output = signal.filtfilt(b, a, ecg_lead)
    d, c = signal.butter(5, w, 'high')
    ecg_lead = signal.filtfilt(d, c, output)
    return ecg_lead

def Scaler(ecg_lead):
    """[The ECG signal is randomly stretched and compressed.
     This is to o create greater diversity in the training data and 
     generate a better generalizability of the model.]

    Args:
        ecg_lead ([list of numpy-Arrays]): [ECG signal]

    Returns:
        ecg_lead [list of numpy-Arrays]: [ECG signal]
    """
  

    number = np.random.uniform(low=0.0, high=1.0, size=None) #Generate a random number between 0 and 1
    ecg_lead = ecg_lead * number #ECG signal is stretched / compressed
    return ecg_lead



def Preprocessing(ecg_leads,ecg_labels,fs,ecg_names,modelname,bin):
    """[Preprocessing of the data. The ECG signal is normalized and cut into pieces of length 600.
        These pieces contain 2 heartbeats each and are added to train_samples. 
        For this purpose, a category is assigned to each heartbeat pair in train_labels]

    Args:
        ecg_leads ([list of numpy-Arrays]): [ECG signal]
        ecg_labels ([list]): [description]
        fs ([int]): [Sampling Frequency]
        ecg_names ([list]): [description]
        modelname ([String]): [description]
        bin ([String]): [description]

    Returns:
        train_labels[list]: [description]
        train_samples [list]: [description]
        r_peaks_list [list]: [description]
        X_train [numpy array]: [description]
        y_train [numpy array]: [description]
        X_test [numpy array]: [description]
        y_test [numpy array]: [description]
    """


    detectors = Detectors(fs)                       # Initialization of the QRS detector
    train_labels = []
    train_samples = []
    r_peaks_list = []
    bin = bin

    line_count = 0
    if (bin=="True"):
        print("Start Binary Data Preprocessing")
        for idx, ecg_lead in enumerate(ecg_leads):
            ecg_lead = ecg_lead.astype('float')  # Change data from Int to Float32 format for various models later on
            ecg_lead = (ecg_lead - ecg_lead.mean()) 
            ecg_lead = ecg_lead / (ecg_lead.std() + 1e-08)  
            r_peaks = detectors.hamilton_detector(ecg_lead)    # Detection of the QRS complexes
            if ecg_labels[idx] == 'N' or ecg_labels[idx] == 'A': #Since binary classifier, only all EGK signals with "A" and "N" are considered
                for r_peak in r_peaks:
                    if r_peak > 150 and r_peak + 450 <= len(ecg_lead): #Testing that you don't get too close to the limits
                        train_samples.append(ecg_lead[r_peak - 150:r_peak + 450])  #Single heartbeats are separated and stored as training data of length 600
                        train_labels.append(ecg_labels[idx])

            line_count = line_count + 1
            if (line_count % 100)==0:
                print(f"{line_count} Dateien wurden verarbeitet.")
            if line_count == 200:  #For test purposes, less data can be used here.
                #break
                pass

        tf.keras.layers.Softmax(axis=-1)
        if(modelname=="CNN" or modelname=="LSTM" or modelname=="Resnet" or modelname=="XGboost"): 
            # Convert classes to one-hot-encoding for CNN, LSTM and Resnet
            # 'N' --> Class 0
            # 'A' --> Class 1
            train_labels = [0 if train_label == 'N' else train_label for train_label in train_labels]
            train_labels = [1 if train_label == 'A' else train_label for train_label in train_labels] 

            if not (modelname == "XGboost"):
                train_labels = keras.utils.to_categorical(train_labels)
                print("One Hot encoding successful")
            else: 
                pass
        elif(modelname=="RandomForrest"):
            pass

    elif (bin=="False"):
        print("Start Multilabel Data Preprocessing")
        for idx, ecg_lead in enumerate(ecg_leads):
            ecg_lead = ecg_lead.astype('float')  # Change data from Int to Float32 format for various models later on
            ecg_lead = (ecg_lead - ecg_lead.mean()) 
            ecg_lead = ecg_lead / (ecg_lead.std() + 1e-08) 
            r_peaks = detectors.hamilton_detector(ecg_lead)     # Detection of the QRS complexes
            for r_peak in r_peaks:
                if r_peak > 150 and r_peak + 450 <= len(ecg_lead):
                    train_samples.append(ecg_lead[r_peak - 150:r_peak + 450]) #Single heartbeats are separated and stored as training data of length 600
                    train_labels.append(ecg_labels[idx])

            line_count = line_count + 1
            if (line_count % 100)==0:
                print(f"{line_count} Dateien wurden verarbeitet.")
            
            if line_count == 100:     #For test purposes, less data can be used here.
                #break
                pass


        if(modelname=="CNN" or modelname=="LSTM"  or modelname=="Resnet" or modelname=="XGboost"):
            # Convert classes to one-hot-encoding for CNN, LSTM and Resnet
            # 'N' --> Class 0
            # 'A' --> Class 1
            # 'O' --> Class 2
            # '~' --> Class 3
            train_labels = [0 if train_label == 'N' else train_label for train_label in train_labels]
            train_labels = [1 if train_label == 'A' else train_label for train_label in train_labels]
            train_labels = [2 if train_label == 'O' else train_label for train_label in train_labels]
            train_labels = [3 if train_label == '~' else train_label for train_label in train_labels]

            if not (modelname == "XGboost"):
                train_labels = keras.utils.to_categorical(train_labels)
            else:
                pass
        elif(modelname=="RandomForrest"):
            pass

    X_train, X_test, y_train, y_test = train_test_split(train_samples, train_labels, test_size=0.2) #Splitting Data 1/5
  
    #Converting Data into Numpy Arrays and reshaping it
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
    X_train = X_train.reshape((*X_train.shape, 1))
    X_test = X_test.reshape((*X_test.shape, 1))


    return (train_labels,train_samples,r_peaks_list,X_train, y_train, X_test, y_test)

