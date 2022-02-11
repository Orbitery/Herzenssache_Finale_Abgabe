from keras.models import load_model
import joblib

def modelload(is_binary_classifier,model_name):  
    """[Distinguishing the model with following loading of the pretrained model]

    Args:
        is_binary_classifier (bool): [shows if binary (True) or multilabel (False) classification is active]
        model_name ([string]): [name of model]

    Returns:
        [model]: [ML Model]
    """

    if(is_binary_classifier==True):
        if (model_name == "CNN"):
            model = load_model("./CNN_bin/model_bin.hdf5")
        elif(model_name=="LSTM"):
            model = load_model("./LSTM_bin/model_bin.hdf5")
        elif(model_name=="Resnet"):
            model = load_model("./Resnet_bin/model_bin.hdf5")            
        elif (model_name=="RandomForrest"):
            model = joblib.load("./RF_bin/random_forest.joblib")
        elif(model_name=="XGboost"):
            try:
                model = joblib.load("./xg_bin/xgboost.joblib")
            except:
                print("Loading of XGBoost Model failed")
    elif(is_binary_classifier ==False):
        if (model_name == "CNN"):
            model = load_model("./CNN_multi/model_multi.hdf5")
        elif(model_name=="LSTM"):
            model = load_model("./LSTM_multi/model_multi.hdf5")
        elif (model_name=="RandomForrest"):
            model = joblib.load("./RF_multi/random_forest.joblib")
        elif(model_name=="Resnet"):
            model = load_model("./Resnet_multi/model_multi.hdf5")     
        elif(model_name=="XGboost"):
            try:
                model = joblib.load("./xg_multi/xgboost.joblib")
            except:
                print("Loading of XGBoost Model failed")


    print ("Model loaded")
    return (model)