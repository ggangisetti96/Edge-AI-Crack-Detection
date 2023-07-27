# -*- coding: utf-8 -*-
"""train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Mf6OgeuQ7mP0bLaC7_-sZmhsaGbu7KXS
"""

!pip install tflite_model_maker

from google.colab import drive
drive.mount('/content/drive')

dir = '/content/drive/MyDrive/Cracks_Dataset/Full_Dataset'

from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader

# Load input data specific to an on-device ML app.
data = DataLoader.from_folder(dir)
train_data, test_data = data.split(0.9)

# Customize the TensorFlow model.
model = image_classifier.create(train_data)

# Evaluate the model.
loss, accuracy = model.evaluate(test_data)

# Export to Tensorflow Lite model and label file in `export_dir`.
model.export(export_dir='/tmp/')

model.export("coral-tflite")

model.evaluate_tflite('coral-tflite/model.tflite', test_data)