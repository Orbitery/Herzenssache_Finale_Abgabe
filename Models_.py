from re import M
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import metrics
import xgboost

def CNN_Model(X_train, y_train, X_test, y_test, epochs_, batchsize):
    """[CNN Model - The architecture was based on the paper "ECG Heartbeat Classification Using Convolutional Neural Networks" by Xu and Liu, 2020]

    Args:
        X_train ([numpy array]): [Numpy array, which contains the ECG signals of the heartbeat pairs for training purposes]
        y_train ([numpy array]): [Numpy array, which contains the classication of the ECG signals of the heartbeat pairs for training purposes]
        X_test ([numpy array]): [Numpy array, which contains the ECG signals of the heartbeat pairs for testing purposes]
        y_test ([numpy array]): [Numpy array, which contains the classication of the ECG signals of the heartbeat pairs for testing purposes]
        epochs_ ([int]): [The number of epochs that the model is to be trained]
        batchsize ([int]): [The batch size with which the model trains]


    Returns:
        model [keras object]: [Contains the trained model]
        history [keras.callbacks.History object]: [Contains values accuracy, validation-accuracy, validation-loss and loss values during the training of the model]    
    """
    print("CNN Model was chosen")
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model = models.Sequential()
    model.add(layers.GaussianNoise(0.1))
    model.add(layers.Conv1D(64, 5, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(layers.Conv1D(64, 5, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv1D(128, 3, activation='relu'))
    model.add(layers.Conv1D(128, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.1))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer='adam',
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=epochs_, batch_size=batchsize,validation_data=(X_test, y_test), callbacks=[callback])

    score = model.evaluate(X_test, y_test)
    print("Accuracy Score: "+str(round(score[1],4)))
    # list all data in 
    print(history.history)
    print(history.history.keys())
    return (model, history)

def LSTM(X_train, y_train, X_test, y_test, epochs_, batchsize):
    """[Simple LSTM network with three layers]

    Args:
        X_train ([numpy array]): [Numpy array, which contains the ECG signals of the heartbeat pairs for training purposes]
        y_train ([numpy array]): [Numpy array, which contains the classication of the ECG signals of the heartbeat pairs for training purposes]
        X_test ([numpy array]): [Numpy array, which contains the ECG signals of the heartbeat pairs for testing purposes]
        y_test ([numpy array]): [Numpy array, which contains the classication of the ECG signals of the heartbeat pairs for testing purposes]
        epochs_ ([int]): [The number of epochs that the model is to be trained]
        batchsize ([int]): [The batch size with which the model trains]

    Returns:
        model [keras object]: [Contains the trained model]
        history [keras.callbacks.History object]: [Contains values accuracy, validation-accuracy, validation-loss and loss values during the training of the model]    
    """


    print("LSTM Model was chosen")
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model = models.Sequential()
    model.add(tf.keras.layers.LSTM(32, return_sequences=True, stateful=False, input_shape = X_train[0].shape))
    model.add(tf.keras.layers.LSTM(20))
    model.add(tf.keras.layers.Dense(y_train.shape[1], activation='softmax'))
    model.compile(optimizer='adam',
                loss=tf.keras.losses.binary_crossentropy,
                metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=epochs_, batch_size=batchsize,validation_data=(X_test, y_test), callbacks=[callback])
    model.build()
    model.summary()
    score = model.evaluate(X_test, y_test)
    print("Accuracy Score: "+str(round(score[1],4)))
    # list all data in 
    print(history.history)
    print(history.history.keys())
    return (model, history)

def RandomForrest(X_train, y_train, X_test, y_test, epochs_, batchsize, treesize_):
    """[Function for the Random Forrest Classifier, which loads the Model]

    Args:
        X_train ([numpy array]): [Numpy array, which contains the ECG signals of the heartbeat pairs for training purposes]
        y_train ([numpy array]): [Numpy array, which contains the classication of the ECG signals of the heartbeat pairs for training purposes]
        X_test ([numpy array]): [Numpy array, which contains the ECG signals of the heartbeat pairs for testing purposes]
        y_test ([numpy array]): [Numpy array, which contains the classication of the ECG signals of the heartbeat pairs for testing purposes]
        epochs_ ([int]): [The number of epochs that the model is to be trained]
        batchsize ([int]): [The batch size with which the model trains]

    Returns:
        m [RandomForrestClassifier object]: [Contains the trained model]
        history [RandomForrestClassifier.History object]: [Contains values accuracy, validation-accuracy, validation-loss and loss values during the training of the model]    
    """

    print("Random Forrest Model was chosen")
    X_train = X_train[:,:,0]
    X_test = X_test[:,:,0]

    #cross validation
    m = RandomForestClassifier(n_jobs=treesize_)
    history = m.fit(X_train, y_train)
    loss = metrics.log_loss(y_test,m.predict_proba(X_test))
    print("Loss is {}".format(loss))
    accuracy = metrics.accuracy_score(y_test,m.predict(X_test))
    print ("Acc is {}".format(accuracy))
    return (m, history)

def Resnet(X_train, y_train, X_test, y_test, epochs_, batchsize):
    """[ResNET Architecture. The idea for this came from the following article.https://towardsdatascience.com/using-resnet-for-time-series-data-4ced1f5395e3, Access: 11.02.2022 19:38 
     And the code of the ResNET layer was largely taken from the following source https://github.com/spdrnl/ecg/blob/master/ECG.ipynb, Access: 11.02.2022 19:38]
    
    The model consists of:

    1x Conv layer
    5x Residual blocks
    1x Flatten layer
    3x Dense layer


    Args:
        X_train ([numpy array]): [Numpy array, which contains the ECG signals of the heartbeat pairs for training purposes]
        y_train ([numpy array]): [Numpy array, which contains the classication of the ECG signals of the heartbeat pairs for training purposes]
        X_test ([numpy array]): [Numpy array, which contains the ECG signals of the heartbeat pairs for testing purposes]
        y_test ([numpy array]): [Numpy array, which contains the classication of the ECG signals of the heartbeat pairs for testing purposes]
        epochs_ ([int]): [The number of epochs that the model is to be trained]
        batchsize ([int]): [The batch size with which the model trains]

    Returns:
        model [keras object]: [Contains the trained model]
        history [keras.callbacks.History object]: [Contains values accuracy, validation-accuracy, validation-loss and loss values during the training of the model] 
    """
    print("Resnet Model was chosen")
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    def get_resnet_model():
        """[Function to call the model]
            Args:
                X_train ([numpy array]): [Numpy array, which contains the ECG signals of the heartbeat pairs for training purposes]
                y_train ([numpy array]): [Numpy array, which contains the classication of the ECG signals of the heartbeat pairs for training purposes]
                X_test ([numpy array]): [Numpy array, which contains the ECG signals of the heartbeat pairs for testing purposes]
                y_test ([numpy array]): [Numpy array, which contains the classication of the ECG signals of the heartbeat pairs for testing purposes]
                epochs_ ([int]): [The number of epochs that the model is to be trained]
                batchsize ([int]): [The batch size with which the model trains]

            Returns:
                model [keras object]: [Contains the trained model]
                history [keras.callbacks.History object]: [Contains values accuracy, validation-accuracy, validation-loss and loss values during the training of the model] 

        """
        def residual_block(X, kernels, stride):
            """[Residual block to more easily connect the layers one after the other] """
            out = tf.keras.layers.Conv1D(32, 5, padding='same')(X)
            out = tf.keras.layers.ReLU()(out)
            out = tf.keras.layers.Conv1D(32, 5, padding='same')(out)
            out = tf.keras.layers.add([X, out])
            out = tf.keras.layers.ReLU()(out)
            out = tf.keras.layers.MaxPool1D(5, 2)(out)
            return out

        kernels = 32
        stride = 5

        inputs = tf.keras.layers.Input(shape=(X_train.shape[1],1))
        X = tf.keras.layers.Conv1D(32, 5)(inputs)
        X = residual_block(X, 32, 5)
        X = residual_block(X, 32, 5)
        X = residual_block(X, 32, 5)
        X = residual_block(X, 32, 5)
        X = residual_block(X, 32, 5)
        X = tf.keras.layers.Flatten()(X)
        X = tf.keras.layers.Dense(32, activation='relu')(X)
        X = tf.keras.layers.Dense(32, activation='relu')(X)
        output = tf.keras.layers.Dense(y_train.shape[1], activation='softmax')(X)

        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model

    model = get_resnet_model() 
    model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])


    history = model.fit(X_train, y_train, epochs=epochs_,validation_data=(X_test, y_test), batch_size=batchsize, callbacks=[callback])
    model.build(input_shape=(X_train.shape[1],1))
    model.summary()
    score = model.evaluate(X_test, y_test)
    print("Accuracy Score: "+str(round(score[1],4)))
    return (model, history)

def XGBoost(X_train, y_train, X_test, y_test, epochs_, batchsize):
    """[Simple XGBoost Model]

    Args:
        X_train ([numpy array]): [Numpy array, which contains the ECG signals of the heartbeat pairs for training purposes]
        y_train ([numpy array]): [Numpy array, which contains the classication of the ECG signals of the heartbeat pairs for training purposes]
        X_test ([numpy array]): [Numpy array, which contains the ECG signals of the heartbeat pairs for testing purposes]
        y_test ([numpy array]): [Numpy array, which contains the classication of the ECG signals of the heartbeat pairs for testing purposes]
        epochs_ ([int]): [The number of epochs that the model is to be trained]
        batchsize ([int]): [The batch size with which the model trains]

    Returns:
        m [XGBClassifier object]: [Contains the trained model]
        history [XGBClassifier.History object]: [Contains values accuracy, validation-accuracy, validation-loss and loss values during the training of the model]    
    """

    print("XGBoost Model was chosen")
    X_train = X_train[:,:,0]
    model = xgboost.XGBClassifier()
    history = model.fit(X_train, y_train)
    return (model, history)
