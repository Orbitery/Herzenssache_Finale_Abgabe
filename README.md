# Herzenssache_Abgabe
Dieses Repository enthält den Code für die automatische Erkennung von Vorhofflimmern aus kurzen EKG-Segmenten mittels Deep Learning. Diese Arbeit wurde im Rahmen des Wettbewerbs "Wettbewerb künstliche Intelligenz in der Medizin" an der TU Darmstadt (KIS*MED, Prof. Hoog Antink) durchgeführt.

## Erste Schritte
Die erforderlichen packages können aus der [`requirements.txt`](https://github.com/Orbitery/Herzenssache_Finale_Abgabe/blob/main/requirements.txt) Datei entnommen werden.

## Funktionen

Insgesamt bietet unser Code folgende Modelle zur Auswahl:

- CNN
- LSTM
- ResNet
- Random Forrest
- XGBoost 

Vor jedem Modell lässt sich optional eine Hauptkomponentenanalyse davorschalten, um die Features zu reduzieren und ggfs. die Trainingszeit zu verringern.  

CNN:
Das CNN wurde nach der Architektur von Xuexiang Xuand und Hongxing Liu [3] entwickelt.  

Resnet:
Das Resnet wurde nach der Idee von Sanne de Roever [1] und mit Teilen des Codes von [3] entwickelt.

Binäres Problem:
- python `predict_pretrained.py` `--model_name` `Resnet` `--is_binary_classifier` `True`

Multi-Class Problem:
- python `predict_pretrained.py` `--model_name` `Resnet` `--is_binary_classifier` `False`


Für ein erfolgreiches benutzerdefiniertes Training wird die Verwendung des Trainingsskripts train.py empfohlen. Hierfür werden folgende Befehle benötigt:

| Argument | Default  Value | Info |
| --- | --- | --- |
| `--modelname` | Resnet | Auswahl von verschiedenen Modellen anhand von Modellname. |
| `--bin` | True | Binäre Darstellung. Unterscheidung zwischen binärer Klassifizierer und Multilabel. |
| `--pca_active` | False | Binäre Darstellung. Option, ob Hauptkomponentenanalyse verwendet wird oder nicht. |
| `--epochs` | 10 | Anzahl von Epochen beim Traininieren des Modells. |
| `--batch_size` | 512 | Gibt die Batchsize zum Trainieren des Modells an. |
| `--treesize` | 50 | Gibt die Anzahl der Bäume des RandomForrest an |


Die Dateien
 - [`predict_pretrained.pyy`](https://github.com/Orbitery/Herzenssache_Finale_Abgabe/blob/main/predict_pretrained.py)
 - [`wettbewerb.py`](https://github.com/Orbitery/Herzenssache_Finale_Abgabe/blob/main/wettbewerb.py)
 - [`score.py`](https://github.com/Orbitery/Herzenssache_Finale_Abgabe/blob/score.py)

stammen aus dem Repository [18-ha-2010-pj](https://github.com/KISMED-TUDa/18-ha-2010-pj) von [Maurice Rohr](https://github.com/MauriceRohr) und [Prof. Hoog Antink](https://github.com/hogius). Die Funktion `predict_labels` in [`predict.py`](https://github.com/Orbitery/Herzenssache_Finale_Abgabe/blob/main/predict.py) beinhaltet das folgende Interface, welches für die Evaluierung verwendet wird.

`predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : str='model.npy',is_binary_classifier : bool=False) -> List[Tuple[str,str]]`

In `model_name` sind die Modelle CNN, LSTM, Random Forest, XGBoost & ResNet enthalten.

## Daten

Die Daten für das Training so wie die Auswertung der Modelle wurden aus dem Repository [18-ha-2010-pj](https://github.com/KISMED-TUDa/18-ha-2010-pj) von 
[Maurice Rohr](https://github.com/MauriceRohr) und [Prof. Hoog Antink](https://github.com/hogius) verwendet. Weitere Trainingsdaten stammen aus dem PTB-XL-EKG-Datensatz, welche von Wissenschaftlerinnen und Wissenschaftler des Fraunhofer Heinrich-Hertz-Instituts (HHI) und der Physikalisch-Technischen Bundesanstalt (PTB) [hier](https://www.physionet.org/content/ptb-xl/1.0.1/) veröffentlich wurden. Ferner besteht die Möglichkeit den Icentia11k [hier](https://academictorrents.com/details/af04abfe9a3c96b30e5dd029eb185e19a7055272) zum Training zu nutzen. 
Für die vortrainierten Modelle wurde ein zusammengesetzter Trainingsdatensatz, bestehend aus den gegebenen Trainingsdaten der Challenge und des PTB-XL-EKG-Datensatzes erstellt. 

## Verweise

```
[1] Resnet Ansatz: {De Roever 2020,
        Titel={{Using ResNet for ECG time-series data}},
        Autor={Sanne de Roever},
        Veröffentlichung={https://towardsdatascience.com/using-resnet-for-time-series-data-4ced1f5395e3, Aufruf:12.01.2022},
        Jahr={2020},
}
```

```
[2] Resnet Architektur: {De Roever 2020,
        Titel={{Replication study of "ECG Heartbeat Classification: A Deep Transferable Representation}},
        Autor={Sanne de Roever},
        Veröffentlichung={https://github.com/spdrnl/ecg/blob/master/ECG.ipynb, Aufruf:12.01.2022},
        Jahr={2020}
}
```

```
[3] CNN Ansatz: {Liu 2017,
        Titel={{ECG Heartbeat Classification UsingConvolutional Neural Networks}},
        Autor={Xuexiang Xuand, Hongxing Liu},
        Veröffentlichung={IEEE Access (Volume:8)},
        Seiten={8614-8619},
        Jahr={2020},
        Organisation={IEEE}
}
```
