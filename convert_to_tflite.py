import tensorflow as tf
from tensorflow.keras.metrics import top_k_categorical_accuracy

# Define custom metrics
def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

# Load the model with custom metrics
model = tf.keras.models.load_model(
    'mobilenet_skin_model_final.keras',
    custom_objects={
        'top_2_accuracy': top_2_accuracy,
        'top_3_accuracy': top_3_accuracy
    }
)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model to a file
with open('mobilenet_skin_model_final.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted to TFLite format successfully!")
