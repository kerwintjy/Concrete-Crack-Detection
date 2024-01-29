import gradio as gr
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('saved_model')


def detect_crack(img):
    img = tf.image.resize(img, (227, 227))  # Image size used for training model
    img = img / 255.0  # Normalizing as per code
    img = np.expand_dims(img, axis=0)  # Expanding dimensions to match the model input shape

    prediction = model.predict(img)

    if prediction > 0.5:
        return "Crack detected"
    else:
        return "No crack detected"


description = ("Concrete crack detection demo. This model is built on the ResNet50v2 architecture. "
               "The model predicts and classifies if an image contains a crack in the concrete."
               "Author: Kerwin Tan")

Title = "Concrete Crack Detector"

iface = gr.Interface(fn=detect_crack, inputs="image", outputs="text", title=Title, description=description)

iface.launch()
