import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('Muscle_Sensor_AI_Project/lstm_model.keras')

# 0,1e-05,1e-05,0,-4e-05,1e-05,-2e-05,0                             actual_class = 1,        predicted_class = 0
# 1e-04,0.00041,5e-05,0.00037,-1e-05,-6e-05,4e-05,0                 actual_class = 2,        predicted_class = 0
# -6e-05,-3e-05,-1e-05,4e-05,4e-05,-3e-05,-0.00017,-7e-05           actual_class = 3,        predicted_class = 3
# 3e-05,2e-05,1e-04,1e-05,0.00019,0.00012,6e-05,2e-05               actual_class = 4,        predicted_class = 4
# 0.00013,0.00012,0.00012,0.00025,0.00035,0.00043,3e-04,0.00018     actual_class = 5,        predicted_class = 5
# 0.00049,0.00017,-7e-05,0.00018,0.00028,0,-0.00011,4e-04           actual_class = 6,        predicted_class = 6
# 1e-05,1e-05,0,0.00025,0.00014,6e-05,-2e-05,-0.00011               actual_class = 7,        predicted_class = 0
# 1e-05,-2e-05,-1e-05,-3e-05,0,-1e-05,0,-1e-05                      actual_class = 0,        predicted_class = 0

# Test loss : 33.34137499332428 %
# Test accuracy : 90.83592891693115 %

def get_user_inputs():
    inputs = []
    print("Please enter 8 input values:")
    for i in range(8):
        value = float(input(f"Enter input {i + 1}: "))
        inputs.append(value)
    return np.array(inputs)

user_inputs = np.array(get_user_inputs())

mean = np.array([-7.95881516e-06, -9.38854240e-06, -9.54846807e-06, -9.60116508e-06, 
                 -1.60469748e-05, -1.08046102e-05, -9.41372582e-06, -9.66059596e-06])

std = np.array([0.00016315, 0.00011924, 0.00012407, 0.00022574, 0.0002724,  
                0.00021517, 0.00015284, 0.0001722])

normalized_inputs = (user_inputs - mean) / std

normalized_inputs = normalized_inputs.reshape(-1, 1, 8)

prediction = model.predict(normalized_inputs)

predicted_class = np.argmax(prediction)

print("Predicted Class:", predicted_class)