# Convert saved model to TFLite
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("./best_model")
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)