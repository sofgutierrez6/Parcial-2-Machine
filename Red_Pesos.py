#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from os import scandir
import pyedflib as pyedf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PyAstronomy import pyaC
import warnings
import scipy as sp
import scipy.fftpack
warnings.filterwarnings('ignore')
#%%
etiquetas_np = np.load('./etiquetas.npy')
descriptores_np = np.load('./descriptores.npy')
#%%
etiquetas_cat = to_categorical(etiquetas_np, 4)
descriptores_np = descriptores_np.transpose()
minimo = descriptores_np.min(axis=0)
maximo = descriptores_np.max(axis=0)
#media = descriptores_np.mean(axis=0)
#desv = descriptores_np.std(axis=0)
descriptores_norm = ((descriptores_np - minimo)/(maximo-minimo))

#dat_train, dat_prueba, etiq_train_1, etiq_prueba_1 = train_test_split(datos, etiquetas_np, test_size=0.2, random_state=4)
#dat_test, dat_val, y_test_1, y_val_1 = train_test_split(X_prueba, y_prueba_1, test_size=0.5, random_state=4)

des_train, des_prueba, etiq_train, etiq_prueba = train_test_split(descriptores_norm, etiquetas_cat, test_size=0.2, random_state=4)
des_test, des_val, etiq_test, etiq_val = train_test_split(des_prueba, etiq_prueba, test_size=0.5, random_state=4)

#%%
des_train_rec = des_train[0:len(des_train)-4,:]
etiq_train_rec = etiq_train[0:len(etiq_train)-4]
#%%
funcion_activacion = 'relu'
initializer = tf.keras.initializers.glorot_uniform(seed=0)
# Creación de la red
modelo_weights = tf.keras.Sequential()
modelo_weights.add(layers.Dense(750, input_shape = (des_train_rec.shape[1],),
                  activation = funcion_activacion))
modelo_weights.add(Dropout(rate=0.5))
modelo_weights.add(layers.Dense(500, 
                  activation = funcion_activacion))
modelo_weights.add(Dropout(rate=0.5))
modelo_weights.add(layers.Dense(250,
                  activation = funcion_activacion))
modelo_weights.add(Dropout(rate=0.5))
modelo_weights.add(layers.Dense(50, 
                  activation = funcion_activacion))
modelo_weights.add(Dropout(rate=0.5))
modelo_weights.add(layers.Dense(4, activation = "softmax"))
Optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
Loss = tf.keras.losses.CategoricalCrossentropy()
modelo_weights.compile(optimizer=Optimizer,
               loss=Loss,
               metrics=['accuracy'])
modelo_weights.summary()
#%%
weights = {0:0.01, 1:0.01, 2:1, 3:0.01}
history_w = modelo_weights.fit(x=des_train_rec,
                     y=etiq_train_rec,
                     class_weight=weights,
                     shuffle=True,
                     steps_per_epoch=100,
                     epochs=200,
                     validation_data=(des_val, etiq_val),
                     initial_epoch=0)

#%%
prediccion_test_w = modelo_weights.predict(x=des_test)
etiquetas_test_w = np.argmax(etiq_test, axis=1)
etiquetas_prediccion_w = np.argmax(prediccion_test_w, axis=1)
cnf_matrix_w = confusion_matrix(etiquetas_test_w, etiquetas_prediccion_w)
print("Matriz de confusión")
print(cnf_matrix_w)

#%%
# Guardar el Modelo
modelo_weights.save('./modelo_pesado.h5')
# Recrea exactamente el mismo modelo solo desde el archivo
# new_model = tf.keras.models.load_model('path_to_my_model.h5')
#%%
acc = history_w.history['accuracy']
val_acc = history_w.history['val_accuracy']

loss = history_w.history['loss']
val_loss = history_w.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1.05])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0.0,1.8])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()