import tensorflow as tf
import numpy as np

@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 1, 8], dtype=tf.float32, name="input"),
    tf.TensorSpec(shape=[None, 8], dtype=tf.float32, name="labels")
])
def train_step(input, labels):
    with tf.GradientTape() as tape:
        logits = model(input, training=True)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, logits))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return {"loss": loss}

mean = np.array([-7.95881516e-06, -9.38854240e-06, -9.54846807e-06, -9.60116508e-06, 
                 -1.60469748e-05, -1.08046102e-05, -9.41372582e-06, -9.66059596e-06])

std = np.array([0.00016315, 0.00011924, 0.00012407, 0.00022574, 0.0002724,  
                0.00021517, 0.00015284, 0.0001722])

@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 1, 8], dtype=tf.float32, name="input")
])
def serve(input):
    input = (input - mean) / std
    return {"output": model(input)}

model = tf.keras.models.load_model('emg_3.keras')

optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-3)

signatures = {
    'serving_default': serve.get_concrete_function(),
    'train': train_step.get_concrete_function()
}

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter.experimental_enable_resource_variables = True

converter.signatures = signatures

tflite_model = converter.convert()

with open('model_with_train_sig.tflite', 'wb') as f:
    f.write(tflite_model)