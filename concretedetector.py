import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model_path = r'C:\Users\ForTh\Downloads\School & Work\NTU\Y3S2\FYP\Crack Detection\code\project\saved_model\resnet50v2_model'
model = tf.keras.models.load_model(model_path)


# Predictive Function
def predict_crack(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (227, 227))  # Assuming this size from your provided code
    img = img / 255.0  # Normalizing as per code
    img = np.expand_dims(img, axis=0)  # Expanding dimensions to match the model input shape

    prediction = model.predict(img)
    return prediction


# GUI
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((250, 250))
        img = ImageTk.PhotoImage(img)
        lbl_img.config(image=img)
        lbl_img.photo = img

        prediction = predict_crack(file_path)
        if prediction > 0.5:
            result_text.set("Crack Detected!")
        else:
            result_text.set("No Crack Detected!")
    else:
        result_text.set("")


# Main Tkinter app
root = tk.Tk()
root.title("Crack Detector")
root.geometry("300x400")

btn_upload = tk.Button(root, text="Upload Image", command=upload_image)
btn_upload.pack(pady=20)

lbl_img = tk.Label(root)
lbl_img.pack()

result_text = tk.StringVar()
lbl_result = tk.Label(root, textvariable=result_text, font=("Arial", 16))
lbl_result.pack(pady=20)

root.mainloop()
