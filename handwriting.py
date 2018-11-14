# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 16:18:24 2018

@author: alexl
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import gzip

def import_images(file_name, num_images, start_index):
    f = gzip.open(file_name, 'rb')
    file_content = f.read()
    images = []
    for h in range(0, num_images):
        image = []
        for i in range(0,28):
            row = []
            for j in range(0,28):
                row.append(file_content[h*28*28+i*28+j+start_index])
            image.append(row)
        images.append(image)
    images_np = np.asarray(images)
    images_np = images_np/255.0
    return images_np

def import_labels(file_name, start_index):
    g = gzip.open(file_name, 'rb')
    file_content = g.read()
    labels = []
    for content in file_content:
        labels.append(content)
    labels = np.asarray(labels)
    labels = labels[start_index:]
    return labels

train_images = import_images('handwritingTrainImages.gz', 60000, 16)
test_images = import_images('handwritingTestImages.gz', 10000, 16)
train_labels = import_labels('handwritingTrainLabels.gz', 8)
test_labels = import_labels('handwritingTestLabels.gz', 8)


""""""""""""""""""""""""""
model1 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model1.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model1.fit(train_images, train_labels, epochs=5)
test_loss1, test_acc1 = model1.evaluate(test_images, test_labels)
predictions1 = model1.predict(test_images)


""""""""""""""""""""""""""
model2 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(1000, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model2.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model2.fit(train_images, train_labels, epochs=5)
test_loss2, test_acc2 = model2.evaluate(test_images, test_labels)
predictions2 = model2.predict(test_images)


""""""""""""""""""""""""""
model3 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(300, activation=tf.nn.relu),
    keras.layers.Dense(100, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model3.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model3.fit(train_images, train_labels, epochs=5)
test_loss3, test_acc3 = model3.evaluate(test_images, test_labels)
predictions3 = model3.predict(test_images)