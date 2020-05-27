#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, Conv1D, Activation, BatchNormalization
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
etiquetas_np = np.load('./etiquetas_13039.npy')
descriptores_np = np.load('./descriptores_13039.npy')
datos_np = np.load('./datos_13039.npy')
#%%
datos_bien = np.zeros((datos_np.shape[0], 3000, 3))
for i in range(datos_np.shape[0]):
    datos_bien[i,:,:] = datos_np[i,:,:].transpose()

#%%
num_canales = 3
etiquetas_cat = to_categorical(etiquetas_np, 4)
datos_norm = np.zeros(datos_bien.shape)
for i in range(num_canales):
    minimo = datos_bien[:,:,i].min()
    maximo = datos_bien[:,:,i].max()
    datos_norm[:,:,i] = ((datos_bien[:,:,i] - minimo)/(maximo-minimo))
#%%
etiquetas_cat = to_categorical(etiquetas_np, 4)
descriptores_np = descriptores_np.transpose()
minimo = descriptores_np.min(axis=0)
maximo = descriptores_np.max(axis=0)
#media = descriptores_np.mean(axis=0)
#desv = descriptores_np.std(axis=0)
descriptores_norm = ((descriptores_np - minimo)/(maximo-minimo))

#%%
dat_train, dat_prueba, etiq_train_1, etiq_prueba_1 = train_test_split(datos_bien, etiquetas_cat, test_size=10456, random_state=4)
dat_test, dat_val, y_test_1, y_val_1 = train_test_split(dat_prueba, etiq_prueba_1, test_size=0.5, random_state=4)

#%%
des_train, des_prueba, etiq_train, etiq_prueba = train_test_split(descriptores_norm, etiquetas_cat, test_size=10456, random_state=4)
des_test, des_val, etiq_test, etiq_val = train_test_split(des_prueba, etiq_prueba, test_size=0.5, random_state=4)

#%%
funcion_activacion = 'relu'
initializer = tf.keras.initializers.glorot_uniform(seed=0)
SHAPE = (3000, num_canales)
# Creación de la red
modelo_deep = tf.keras.Sequential()
modelo_deep.add(tf.keras.layers.Conv1D(64, 5, strides=3, padding='same', input_shape=SHAPE))
modelo_deep.add(tf.keras.layers.Conv1D(128, 5, strides=2, padding='same'))
modelo_deep.add(Dropout(rate=0.2))
modelo_deep.add(tf.keras.layers.Conv1D(128, 13, strides=1, padding='same'))
modelo_deep.add(tf.keras.layers.Conv1D(256, 7, strides=2, padding='same'))
modelo_deep.add(tf.keras.layers.Conv1D(256, 7, strides=1, padding='same'))
modelo_deep.add(tf.keras.layers.Conv1D(64, 4, strides=2, padding='same'))
modelo_deep.add(tf.keras.layers.Conv1D(32, 3, strides=1, padding='same'))
modelo_deep.add(tf.keras.layers.Conv1D(64, 6, strides=2, padding='same'))
modelo_deep.add(tf.keras.layers.Conv1D(8, 5, strides=1, padding='same'))
modelo_deep.add(tf.keras.layers.Conv1D(8, 2, strides=2, padding='same'))
modelo_deep.add(tf.keras.layers.Flatten())

modelo_deep.add(layers.Dense(750, input_shape = (des_train.shape[1],), activation = funcion_activacion))
modelo_deep.add(Dropout(rate=0.5))
modelo_deep.add(layers.Dense(500, activation = funcion_activacion))
modelo_deep.add(Dropout(rate=0.5))
modelo_deep.add(layers.Dense(250, activation = funcion_activacion))
modelo_deep.add(Dropout(rate=0.5))
modelo_deep.add(layers.Dense(50, activation = funcion_activacion))
modelo_deep.add(Dropout(rate=0.5))
modelo_deep.add(layers.Dense(4, activation = "softmax"))

Optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
Loss = tf.keras.losses.CategoricalCrossentropy()
modelo_deep.compile(optimizer=Optimizer,
               loss=Loss,
               metrics=['accuracy'])
modelo_deep.summary()

#%%
history_deep = modelo_deep.fit(x=dat_train,
                     y=etiq_train,
                     shuffle=True,
                     steps_per_epoch=100,
                     epochs=25,
                     validation_data=(dat_val, etiq_val),
                     initial_epoch=0)

#%%
prediccion_test_deep = modelo_deep.predict(x=des_test)
etiquetas_test_deep = np.argmax(etiq_test, axis=1)
etiquetas_prediccion_deep = np.argmax(prediccion_test_deep, axis=1)
cnf_matrix = confusion_matrix(etiquetas_test_deep, etiquetas_prediccion_deep)
print("Matriz de confusión")
print(cnf_matrix)

#%%
# Guardar el Modelo
modelo_deep.save('./modelo_deep.h5')
# Recrea exactamente el mismo modelo solo desde el archivo
# new_model = tf.keras.models.load_model('path_to_my_model.h5')
#%%
acc = history_deep.history['accuracy']
val_acc = history_deep.history['val_accuracy']

loss = history_deep.history['loss']
val_loss = history_deep.history['val_loss']

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