# -*- coding: utf-8 -*-

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow.keras as keras
from ecgdetectors import Detectors
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models

from wettbewerb import load_references

### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

ecg_leads,ecg_labels,fs,ecg_names = load_references("../training") # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz

detectors = Detectors(fs)                                 # Initialisierung des QRS-Detektors

train_labels = []
train_samples = []
r_peaks_list = []

line_count = 0
for idx, ecg_lead in enumerate(ecg_leads):
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
    if line_count == 500:  #Für Testzwecke kann hier mit weniger Daten gearbeitet werden.
      break
      #pass



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



print(np.array(X_train[0]).shape)
#Definieren der CNN Architektur. Hierbei wurde sich bei der Architektur an dem Paper "ECG Heartbeat Classification Using Convolutional Neural Networks" von Xu und Liu, 2020 orientiert. 
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model = models.Sequential()
model.add(tf.keras.layers.LSTM(32, return_sequences=True, stateful=False, input_shape = X_train[0].shape))
model.add(tf.keras.layers.LSTM(20))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
model.compile(optimizer='adam',
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=1028, callbacks=[callback])
model.build()
model.summary()
score = model.evaluate(X_test, y_test)
print("Accuracy Score: "+str(round(score[1],4)))



