import pandas as pd
import numpy as np
import wfdb
import ast
import csv  
import sys
import numpy
import scipy.io



def load_raw_data(df, sampling_rate, path):
    """[summary]

    Args:
        df ([type]): [description]
        sampling_rate ([type]): [description]
        path ([type]): [description]

    Returns:
        [type]: [description]
    """

    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
        print("1")
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
        print("2")
    data = np.array([signal for signal, meta in data])
    print("3")
    return data

path = 'C:/Users/chris/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
sampling_rate=100

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
print("4")

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)
print("5")

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    """[summary]

    Args:
        y_dic ([type]): [description]

    Returns:
        [type]: [description]
    """

    tmp = []
    for key in y_dic.keys():
        print("6")
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

csv_array=[]

#for i in range(0,21837):
for i in range (0,21837):
    for j in range(0,999):
        csv_array= np.append(csv_array,X[i,j])
        print("---{}.line processed".format(j))
    #np.savetxt("my_data.csv", csv_array, delimiter=",")
    #data1 = [ ]
    #with open("my_data.csv") as f:
    #    reader = csv.reader(f)
    #    for row in reader:
    #        rowData = [ float(elem) for elem in row ]
    #        data1.append(rowData)
    matrix = numpy.array(csv_array)
    matrix = numpy.transpose(matrix)
    number = 6001 + i
    scipy.io.savemat('train_ecg_{}.mat'.format(number), {'val':matrix})
    print("ecg{}_.mat created".format(i))
    csv_array = []
    matrix = []

# Split data into train and test
test_fold = 10
# Train
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# Test
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
