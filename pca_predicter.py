def pca_predict(data_samples):
    """[pre-trained transformation matrix is loaded and used for PCA.]
    Args:
        data_samples [numpy array]: [Numpy array, which contains the ECG signals of the heartbeat pairs for training purposes]

    Returns:
      data_samples_new  [numpy array]: [Processed new Features]
    """
    X_temp = data_samples[:,:,0]

    with open('pca.pkl', 'rb') as pickle_file:
        pca = pickle.load(pickle_file)
    
    pca.fit(X_temp)

    #Transforming training data
    X_new = pca.transform(X_temp)

    #Reshaping
    data_samples_new = X_new.reshape((*X_new.shape,1))

    return data_samples