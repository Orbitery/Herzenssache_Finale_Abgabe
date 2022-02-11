import numpy as np
def decider(predicted, ecg_names,data_samples,data_names, is_binary_classifier):
    """[Prediction results are analyzed. Each ECG signal is assigned a class by majority vote. This classification is then returned.
        e.g. in an ECG signal 50 "A heartbeats" and 2 "O heartbeats" were detected, then the corresponding ECG signal is classified as "A". ]

    Args:
        predicted (list): [List with the probabilites of each class]
        ecg_names (list): [List with the unique name of every ecg-signal]
        data_samples (list): [List with Values of ecg-signal]
        data_names (list): [List with the unique name of the ecg signal, which belongs to the heartbeats]
        is_binary_classifier (bool): [shows if binary (True) or multilabel (False) classification is active]

    Returns:
       predictions [list of tuples]: [Values of ECG-Predictions]
    """

    data_samples = np.array(data_samples)
    data_samples = data_samples.reshape((*data_samples.shape, 1))

    predictions = list()
    label_predicted = []
    label_predicted_democatric = []
    x = 0

    if (is_binary_classifier==True):
        for row in predicted: #Determine the most probable class
            if predicted[x,0]>predicted[x,1]:
                label_predicted.append("N")
            elif predicted[x,0]<predicted[x,1]:
                label_predicted.append("A")
            else:
                print("Error")
            x = x + 1
        n_sum = 0
        a_sum = 0
        t = 0

        for ecg_row in ecg_names: #Democratic approach to classify ECG signals based on heartbeat predictions.
            for idx, y in enumerate(data_names):
                if (ecg_row==y):
                    if (label_predicted[idx]=='N'):
                        n_sum = n_sum + 1
                    elif (label_predicted[idx]=='A'):
                        a_sum = a_sum +1
                else:
                    pass
            if (n_sum>=a_sum):
                label_predicted_democatric.append("N")
            elif (n_sum<a_sum):
                label_predicted_democatric.append("A")
            print("In {}: Number of A-Heartbeats: {}, Number of N-Heartbeats: {}".format(ecg_row,a_sum, n_sum))
            n_sum = 0
            a_sum = 0       
                    


        for idx, name_row in enumerate(ecg_names): #Create the final predictions
            predictions.append((ecg_names[idx], label_predicted_democatric[idx]))

    elif (is_binary_classifier == False):
        for row in predicted:   #Determine the most probable class
            if (((predicted[x,0]>predicted[x,1]) and (predicted[x,0]> predicted[x,2]) and (predicted[x,0]>predicted[x,3]))):
                label_predicted.append("N")
            elif (((predicted[x,1]>predicted[x,0]) and (predicted[x,1]> predicted[x,2]) and (predicted[x,1]>predicted[x,3]))):
                label_predicted.append("A")
            elif (((predicted[x,2]>predicted[x,0]) and (predicted[x,2]> predicted[x,1]) and (predicted[x,2]>predicted[x,3]))):
                label_predicted.append("O")
            elif (((predicted[x,3]>predicted[x,0]) and (predicted[x,3]> predicted[x,1]) and (predicted[x,3]>predicted[x,2]))):
                label_predicted.append("~")
            else:
                print("Error")
            x = x + 1
        n_sum = 0
        a_sum = 0
        o_sum = 0
        t_sum = 0

        t = 0
        for ecg_row in ecg_names:  #Democratic approach to classify ECG signals based on heartbeat predictions.
            for idx, y in enumerate(data_names):
                if (ecg_row==y):
                    if (label_predicted[idx]=='N'):
                        n_sum = n_sum + 1
                    elif (label_predicted[idx]=='A'):
                        a_sum = a_sum +1
                    elif (label_predicted[idx]=='O'):
                        o_sum = o_sum +1    
                    elif (label_predicted[idx]=='~'):
                        o_sum = o_sum +1    
                else:
                    pass
            if ((n_sum>=a_sum)and(n_sum>=o_sum)and(n_sum>=t_sum)):
                label_predicted_democatric.append("N")
            elif ((a_sum>=n_sum)and(a_sum>=o_sum)and(a_sum>=t_sum)):
                label_predicted_democatric.append("A")
            elif ((o_sum>=n_sum)and(o_sum>=a_sum)and(o_sum>=t_sum)):
                label_predicted_democatric.append("O")
            elif ((t_sum>=n_sum)and(t_sum>=o_sum)and(t_sum>=a_sum)):
                label_predicted_democatric.append("~")
            print("In {}: Number of A-Heartbeats: {}, Number of N-Heartbeats: {}, Number of O-Heartbeats: {}, Number of ~-Heartbeats: {}".format(ecg_row,a_sum, n_sum,o_sum,t_sum))
            n_sum = 0
            a_sum = 0
            o_sum = 0
            t_sum = 0

                    
        for idx, name_row in enumerate(ecg_names): #Create the final predictions
            predictions.append((ecg_names[idx], label_predicted_democatric[idx]))
        print("Predictions are created")
    return (predictions)     
        