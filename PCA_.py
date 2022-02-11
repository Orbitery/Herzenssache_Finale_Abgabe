from sklearn.decomposition import PCA
import numpy as np
import pickle

def PCA_function (X_train,X_test,modelname):
        if not (modelname =="Resnet"):
                print("Start PCA Feature Reduction")
                X_temp = X_train[:,:,0]

                # First, do pca with all features
                pca = PCA(n_components=X_temp.shape[1])
                pca.fit(X_temp)
                explained_variance = pca.explained_variance_ratio_

                # Have a look how many new features do we need to explain 99 % of the variance in the original data set
                iw = np.argwhere(np.cumsum(explained_variance)>0.99)[0][0]
                print(f"Max Features explain 99 % of variance is {iw+1}")

                # Do pca with reduced numer of features 
                pca = PCA(n_components=iw+1)
                pca.fit(X_temp)

                #Transforming training data
                X_new = pca.transform(X_temp)

                # Test data can only be transformed with pca fitted with test data
                X_test_new = pca.transform(X_test[:,:,0])

                #Reshaping
                X_new = X_new.reshape((*X_new.shape,1))
                X_test_new = X_test_new .reshape((*X_test_new .shape,1))

                X_train = X_new
                X_test = X_test_new

                with open('./CNN_bin/pca.pkl', 'wb') as pickle_file:
                        pickle.dump(pca, pickle_file)
                        
                return X_train, X_test

        elif (modelname=="Resnet"):
                print("Resnet Architecture can not be used with PCA due to limited input features")
