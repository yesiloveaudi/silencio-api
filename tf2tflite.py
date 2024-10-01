import tensorflow as tf

# Load the TensorFlow model
converter = tf.lite.TFLiteConverter.from_saved_model('model.pb')

# Convert the model to TensorFlow Lite
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
