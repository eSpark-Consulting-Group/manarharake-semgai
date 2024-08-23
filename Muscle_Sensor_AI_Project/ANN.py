import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

dataset=pd.read_csv("EMG-data.csv")

##drop gesture 0 because it offers no information due to its unmarked nature
index_numbers_1=dataset[dataset["class"]==0].index 
dataset.drop(index_numbers_1,inplace=True)
##drop gesture 7 because it offers no information due to it being performed 
##by just two out of 36 patients 
index_numbers_2=dataset[dataset["class"]==7].index
dataset.drop(index_numbers_2,inplace=True)

dataset=dataset.drop(columns=["time"])
dataset["class"].value_counts()
dataset=dataset.groupby(['label','class'])

#functions for extracting sEMG features
def rms(data): ##root mean square
      return  np.sqrt(np.mean(data**2,axis=0))  

def SSI(data): ##Simple Square Integral
    return np.sum(data**2,axis=0)

def abs_diffs_signal(data): ##absolute differential signal
    return np.sum(np.abs(np.diff(data,axis=0)),axis=0)

##function for returning an estimator class name 
def print_estimator_name(estimator):
    return estimator.__class__.__name__


dataset=dataset.agg(['min','max',rms,SSI,abs_diffs_signal,np.ptp])

dataset=dataset.reset_index()

dataset.loc[dataset["class"] == 0, "gesture"] = "unmarked data"
dataset.loc[dataset["class"] == 1, "gesture"] = "hand at rest"
dataset.loc[dataset["class"] == 2, "gesture"] = "hand clenched in a fist"
dataset.loc[dataset["class"] == 3, "gesture"] = "wrist flexion"
dataset.loc[dataset["class"] == 4, "gesture"] = "wrist extension"
dataset.loc[dataset["class"] == 5, "gesture"] = "radial deviations"
dataset.loc[dataset["class"] == 6, "gesture"] = "ulnar deviations"
dataset.loc[dataset["class"] == 7, "gesture"] = "extended palm"

features=dataset.drop(columns=["label","class","gesture"])
labels=dataset["class"]

X_train, X_test,y_train,y_test= train_test_split(features,labels,test_size=0.30, random_state=1)

scaler = StandardScaler()
scaler.fit(X_train)

train_X = scaler.transform(X_train)
test_X = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(50, 50, 50))
mlp.fit(train_X, y_train)

y_pred = mlp.predict(test_X)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

# Converting the model to ONNX format
initial_type = [('float_input', FloatTensorType([None, train_X.shape[1]]))]
onnx_model = convert_sklearn(mlp, initial_types=initial_type)

# Saving the ONNX model to a file
with open("mlp_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())