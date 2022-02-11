import os
import joblib

def Saver(model,bin,modelname):
    """[A distinction is made between the different models and these are saved]

    Args:
        model (object): [ML Model]
        bin (String): [shows if binary ("True") or multilabel ("False") classification is active]
        modelname (String): [Modelname]
    """

    if (modelname =="CNN"):
        if not os.path.exists("./CNN_bin/"):
            os.mkdir("./CNN_bin/")
        else:
            pass
        if not os.path.exists("./CNN_multi/"):
            os.mkdir("./CNN_multi/")
        else:
            pass
        try:    
            with open('./CNN_bin/modelsummary.txt', 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
        except:
            print ("Unable to write summary")

        if (bin=="True"):
            if os.path.exists("./CNN_bin/model_bin.hdf5"):
                os.remove("./CNN_bin/model_bin.hdf5")
            else:
                pass
            model.save("./CNN_bin/model_bin.hdf5")
            print("Binary CNN Model saved")
        elif (bin=="False"):
            if os.path.exists("./CNN_multi/model_multi.hdf5"):
                os.remove("./CNN_multi/model_multi.hdf5")
            else:
                pass
            model.save("./CNN_multi/model_multi.hdf5")
            print("Multilabel CNN Model saved")
    elif(modelname=="LSTM"):
        if not os.path.exists("./LSTM_bin/"):
            os.mkdir("./LSTM_bin/")
        else:
            pass
        if not os.path.exists("./LSTM_multi/"):
            os.mkdir("./LSTM_multi/")
        else:
            pass
        try:    
            with open('./LSTM_bin/modelsummary.txt', 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
        except:
            print ("Unable to write summary")

        if (bin=="True"):
            if os.path.exists("./LSTM_bin/model_bin.hdf5"):
                os.remove("./LSTM_bin/model_bin.hdf5")
            else:
                pass
            print("Binary LSTM Model saved")
            model.save("./LSTM_bin/model_bin.hdf5")

        elif (bin=="False"):
            if os.path.exists("./LSTM_multi/model_multi.hdf5"):
                os.remove("./LSTM_multi/model_multi.hdf5")
            else:
                pass
            model.save("./LSTM_multi/model_multi.hdf5")
            print("Multilabel LSTM Model saved")
    elif(modelname=="RandomForrest"):
        if not os.path.exists("./RF_bin/"):
            os.mkdir("./RF_bin/")
        else:
            pass    
        if not os.path.exists("./RF_multi/"):
            os.mkdir("./RF_multi/")
        else:
            pass
        try:
            with open('./RF_bin/modelsummary.txt', 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
        except:
            print ("Unable to write summary")

        if (bin=="True"):
            if os.path.exists("./RF_bin/random_forest.joblib"):
                os.remove("./RF_bin/random_forest.joblib")
            else:
                pass
            joblib.dump(model, "./RF_bin/random_forest.joblib")
            print("Binary RF Model saved")

        elif (bin=="False"):
            if os.path.exists("./RF_multi/random_forest.joblib"):
                os.remove("./RF_multi/random_forest.joblib")
            else:
                pass
            joblib.dump(model, "./RF_bin/random_forest.joblib")
            print("Multilabel RF Model saved")
    elif(modelname=="Resnet"):
        if not os.path.exists("./Resnet_bin/"):
            os.mkdir("./Resnet_bin/")
        else:
            pass
        if not os.path.exists("./Resnet_multi/"):
            os.mkdir("./Resnet_multi/")
        else:
            pass
        try:    
            with open('./Resnet_bin/modelsummary.txt', 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
        except:
            print ("Unable to write summary")

        if (bin=="True"):
            if os.path.exists("./Resnet_bin/model_bin.hdf5"):
                os.remove("./Resnet_bin/model_bin.hdf5")
            else:
                pass
            model.save("./Resnet_bin/model_bin.hdf5")
            print("Binary Resnet Model saved")

        elif (bin=="False"):
            if os.path.exists("./Resnet_multi/model_multi.hdf5"):
                os.remove("./Resnet_multi/model_multi.hdf5")
            else:
                pass
            model.save("./Resnet_multi/model_multi.hdf5")
            print("Multilabel Resnet Model saved")
    elif(modelname=="XGboost"):
        if not os.path.exists("./xg_bin/"):
            os.mkdir("./xg_bin/")
        else:
            pass    
        if not os.path.exists("./xg_multi/"):
            os.mkdir("./xg_multi/")
        else:
            pass
        try:
            with open('./xg_bin/modelsummary.txt', 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
        except:
            print ("Unable to write summary")

        if (bin=="True"):
            if os.path.exists("./xg_bin/xgboost.joblib"):
                os.remove("./xg_bin/xgboost.joblib")
            else:
                pass
            joblib.dump(model, "./xg_bin/xgboost.joblib")
            print("Binary xgboost Model saved")

        elif (bin=="False"):
            if os.path.exists("./xg_multi/xgboost.joblib"):
                os.remove("./xg_multi/xgboost.joblib")
            else:
                pass
            joblib.dump(model, "./xg_multi/xgboost.joblib")
            print("Multilabel xgboost Model saved")
    