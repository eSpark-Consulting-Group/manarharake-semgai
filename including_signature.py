import tensorflow as tf

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

@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 1, 8], dtype=tf.float32, name="input")
])
def serve(input):
    return {"output": model(input)}

model = tf.keras.models.load_model('cnn_model.keras')

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

with open('cnn_model_with_train_sig.tflite', 'wb') as f:
    f.write(tflite_model)