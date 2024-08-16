import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)


# Reshape X to fit into Conv1D (samples, time steps, features)
# Here we assume each gesture instance is treated as a time step (temporal structure)
n_channels = 8  # Assuming 8 channels as features
time_steps = X_scaled.shape[1] // n_channels
X_reshaped = X_scaled.reshape(X_scaled.shape[0], time_steps, n_channels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, labels, test_size=0.2, random_state=42)

cnn = Sequential([
    Input(shape=(time_steps, n_channels)),
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    MaxPooling1D(pool_size=1),
    Dropout(0.5),
    Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
    MaxPooling1D(pool_size=1),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # Output layer with number of classes
])

# Compile the model
cnn.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',  # Use sparse categorical crossentropy for integer labels
              metrics=['accuracy'])

# Train the model
cnn.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = cnn.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')

predictions = cnn.predict(X_test)

predicted_classes = np.argmax(predictions, axis=1)

accuracy= accuracy_score(y_test, predicted_classes)
print(f'Model accuracy: {accuracy}')

# Convert directly to TFLite instead of saving as .keras
converter = tf.lite.TFLiteConverter.from_keras_model(cnn)
tflite_model = converter.convert()

# Save the .tflite model to a file
with open('cnn_model.tflite', 'wb') as f:
    f.write(tflite_model)

# cnn.save('cnn_model.keras')