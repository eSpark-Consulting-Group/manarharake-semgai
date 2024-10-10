import numpy as np
import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('emg_3.keras')

# Function to get 8 input values from the user
def get_user_inputs():
    inputs = []
    print("Please enter 8 input values:")
    for i in range(8):
        value = float(input(f"Enter input {i + 1}: "))
        inputs.append(value)
    return np.array(inputs)

# Get inputs from the user
user_inputs = get_user_inputs()

# Mean and standard deviation (obtained from training data)
mean = np.array([-7.95881516e-06, -9.38854240e-06, -9.54846807e-06, -9.60116508e-06, 
                 -1.60469748e-05, -1.08046102e-05, -9.41372582e-06, -9.66059596e-06])

std = np.array([0.00016315, 0.00011924, 0.00012407, 0.00022574, 0.0002724,  
                0.00021517, 0.00015284, 0.0001722])

# Normalize the inputs by subtracting the mean and dividing by the standard deviation
normalized_inputs = (user_inputs - mean) / std

# Ensure input shape matches the model's expectation (add batch dimension)
input_data = np.expand_dims(normalized_inputs, axis=0)

# Pass the inputs to the model and get the prediction
prediction = model.predict(input_data)

# Get the predicted class (assuming classification task)
predicted_class = np.argmax(prediction)

# Output the predicted class
print("Predicted Class:", predicted_class)
