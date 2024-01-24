# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, ResNet50V2, ResNet152V2, VGG16, VGG19
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import warnings
warnings.simplefilter("ignore")

# 1. Data Preparation
# Turn the directory paths into Path object
positive_dir = Path(r'C:\Users\ForTh\Downloads\School & Work\NTU\Y3S2\FYP\Crack Detection\test\Positive')
negative_dir = Path(r'C:\Users\ForTh\Downloads\School & Work\NTU\Y3S2\FYP\Crack Detection\test\Negative')

def generate_df(image_dir, label):
    """
    Create the DataFrame of the associated directory and label.
    """

    filepaths = pd.Series(list(image_dir.glob(r'*.jpg')), name='Filepath').astype(str)
    labels = pd.Series(label, name='Label', index=filepaths.index)
    df = pd.concat([filepaths, labels], axis=1)

    return df


positive_df = generate_df(positive_dir, 'POSITIVE')
negative_df = generate_df(negative_dir, 'NEGATIVE')
data = pd.concat([positive_df, negative_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
data.head()

# Split Training and Test sets
train_df, test_df = train_test_split(
    data.sample(6000, random_state=0), # Keep only 6000 samples to save computation time.
    train_size=0.7,
    shuffle=True,
    random_state=42)

# Image generator for the training set
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255,
    validation_split=0.2,
)

# Image generator for the test set
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255
)

# Generate training images
train_images = train_generator.flow_from_dataframe(
    train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(227, 227),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)

# Generate validation images
val_images = train_generator.flow_from_dataframe(
    train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(227, 227),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation'
)

# Generate test images
test_images = test_generator.flow_from_dataframe(
    test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(227, 227),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=False
)

# Build Model
def build_model(base_model):
    base_model.trainable = False
    inputs = Input(shape=(227, 227, 3))
    x = base_model(inputs, training=False)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

# Dict of pre-trained models
model = ResNet50V2

base = model(include_top=False, weights='imagenet')
model = build_model(base)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
# Train and Validate
print(f"Training {model}...")
history = model.fit(train_images, validation_data=val_images, epochs=1,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            min_delta=0.0002,
                            patience=3,
                            mode='min',
                            verbose=1,
                            baseline=None,
                            restore_best_weights=True
                            )
                        ]
                    )

# Saving the ResNet50v2 Model
model_save_path = r"C:\Users\ForTh\Downloads\Comp Projects\Concrete Crack Detection\code\saved_model\resnet50v2_model2"
model.save(model_save_path)

print("Process completed!")
