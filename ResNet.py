#def CNN_Model(X_train, y_train, X_test, y_test):

import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow.keras as keras
from ecgdetectors import Detectors

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


#sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tensorflow.keras import datasets, layers, models


from wettbewerb import load_references

from preprocessing_ import *


### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

ecg_leads,ecg_labels,fs,ecg_names = load_references("../training") # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz

detectors = Detectors(fs)                                 # Initialisierung des QRS-Detektors

train_labels = []
train_samples = []
r_peaks_list = []

line_count = 0
for idx, ecg_lead in enumerate(ecg_leads):
    ecg_lead = BP_Filter(ecg_lead)
    ecg_lead = Scaler(ecg_lead)
    ecg_lead = ecg_lead.astype('float')  # Wandel der Daten von Int in Float32 Format für CNN später
    ecg_lead = (ecg_lead - ecg_lead.mean())
    ecg_lead = ecg_lead / (ecg_lead.std() + 1e-08)
    r_peaks = detectors.hamilton_detector(ecg_lead)     # Detektion der QRS-Komplexe
    if ecg_labels[idx] == 'N' or ecg_labels[idx] == 'A':
        for r_peak in r_peaks:
            if r_peak > 150 and r_peak + 450 <= len(ecg_lead):
              train_samples.append(ecg_lead[r_peak - 150:r_peak + 450]) #Einzelne Herzschläge werden separiert und als Trainingsdaten der Länge 300 abgespeichert
              train_labels.append(ecg_labels[idx])

    line_count = line_count + 1
    if (line_count % 100)==0:
      print(f"{line_count} Dateien wurden verarbeitet.")
    if line_count == 100:  #Für Testzwecke kann hier mit weniger Daten gearbeitet werden.
      break
      #pass

start = time.perf_counter()


tf.keras.layers.Softmax(axis=-1)

# Klassen in one-hot-encoding konvertieren
# 'N' --> Klasse 0
# 'A' --> Klasse 1
train_labels = [0 if train_label == 'N' else train_label for train_label in train_labels]
train_labels = [1 if train_label == 'A' else train_label for train_label in train_labels]
train_labels = keras.utils.to_categorical(train_labels)

X_train, X_test, y_train, y_test = train_test_split(train_samples, train_labels, test_size=0.2)

X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
X_train = X_train.reshape((*X_train.shape, 1))
X_test = X_test.reshape((*X_test.shape, 1))

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)


def get_resnet_model(categories=2):
  def residual_block(X, kernels, stride):
    out = keras.layers.Conv1D(kernels, stride, padding='same')(X)
    out = keras.layers.ReLU()(out)
    out = keras.layers.Conv1D(kernels, stride, padding='same')(out)
    out = keras.layers.add([X, out])
    out = keras.layers.ReLU()(out)
    out = keras.layers.MaxPool1D(5, 2)(out)
    return out

  kernels = 32
  stride = 5

  inputs = keras.layers.Input(shape=(X_train.shape[1],1))
  X = keras.layers.Conv1D(kernels, stride)(inputs)
  X = residual_block(X, kernels, stride)
  X = residual_block(X, kernels, stride)
  X = residual_block(X, kernels, stride)
  X = residual_block(X, kernels, stride)
  X = residual_block(X, kernels, stride)
  X = keras.layers.Flatten()(X)
  X = keras.layers.Dense(32, activation='relu')(X)
  X = keras.layers.Dense(32, activation='relu')(X)
  output = keras.layers.Dense(y_train.shape[1], activation='softmax')(X)

  model = keras.Model(inputs=inputs, outputs=output)
  return model

model = get_resnet_model()
model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=3,validation_data=(X_test, y_test), batch_size=1028, callbacks=[callback])
model.build(input_shape=(X_train.shape[1],1))
model.summary()
score = model.evaluate(X_test, y_test)
print("Accuracy Score: "+str(round(score[1],4)))

if os.path.exists("./resnet_Model/model_bin.hdf5"):
    os.remove("./resnet_Model/model_bin.hdf5")

else:
    pass

if not os.path.exists("./resnet_Model/"):
    os.makedirs("./resnet_Model/")

if os.path.exists("./resnet_Model/model_bin.hdf5"):
    os.remove("./resnet_Model/model_bin.hdf5")

model.save("./resnet_Model/model_bin.hdf5")

with open('./resnet_Model/modelsummary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))


from plotter import *

# list all data in
print(history.history)
print(history.history.keys())

plot_creater(history,True, "resnet")

