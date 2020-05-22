# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
import tensorflow as tf
from sklearn import svm
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
# %%
ruta_archivos = "../../data/sleep-edf-database-expanded-1.0.0/sleep-cassette"
lista_archivos = [arch.name for arch in scandir(ruta_archivos) if arch.is_file()]
# %%
datos = []
etiquetas = []
descriptores = [[] for _ in range(9)]
arreglo_x = np.array(list(range(3000)))
for ruta_actual in lista_archivos:
    if ruta_actual[9:len(ruta_actual)] == "Hypnogram.edf":
        identificador = ruta_actual[2:7]
        
        # Lectura del archivo de estados de sueño (etiquetas)
        st_FileHypEdf = pyedf.EdfReader(ruta_archivos + "/" + ruta_actual)
        # Datos en ventanas de 30 segundos, 
        # v_HypTime es el tiempo de inicio, v_HypDur es la duración en un estado específico (pueden ser varias ventanas),
        # v_Hyp es la etiqueta.
        v_HypTime, v_HypDur, v_Hyp = st_FileHypEdf.readAnnotations()
        
        # Lectura de las señales s_SigNum señales con nombres v_Signal_Labels
        st_FileEdf = pyedf.EdfReader(ruta_archivos + "/SC" + identificador + "0-PSG.edf")
        s_SigNum = st_FileEdf.signals_in_file
        v_Signal_Labels = st_FileEdf.getSignalLabels()
        
        # Conversion a segundos usando frecuencia de muestreo.
        s_SigRef = 0
        s_NSamples = st_FileEdf.getNSamples()[0]
        s_FsHz = st_FileEdf.getSampleFrequency(s_SigRef)
        
        # v_Sig = np.zeros((s_NSamples, 1))
        v_Sig = st_FileEdf.readSignal(s_SigRef)
        v_Time = np.arange(0, s_NSamples) / s_FsHz
        
        s_WinSizeSec = 30
        s_WinSizeSam = np.round(s_FsHz * s_WinSizeSec)
        s_FirstInd = 0
        while 1:
            s_LastInd = s_FirstInd + s_WinSizeSam
            if s_LastInd > s_NSamples:
                break
            for i in range(len(v_Hyp)-1):
                if ((v_HypTime[i] <= s_FirstInd/s_FsHz) and (s_FirstInd/s_FsHz < v_HypTime[i+1])):
                    if v_Hyp[i] != "Sleep stage ?":
                        etiquetas.append(v_Hyp[i])
                        datos.append(np.array(v_Sig[s_FirstInd:s_LastInd]))
                        dato_df = pd.DataFrame(v_Sig[s_FirstInd:s_LastInd])
                        descriptores[0].append(dato_df.mean()[0])
                        descriptores[1].append(dato_df.var()[0])
                        descriptores[2].append(dato_df.kurt()[0])
                        descriptores[3].append(dato_df.skew()[0])
                        descriptores[4].append(len(pyaC.zerocross1d(x=arreglo_x, y=np.array(v_Sig[s_FirstInd:s_LastInd]))))
                        # Frecuencia
                        dato_fft = sp.fftpack.fft(dato_df)
                        fftfreq = sp.fftpack.fftfreq(len(dato_fft), 1/s_FsHz)
                        potencia_02 = 0
                        potencia_28 = 0
                        potencia_814 =  0
                        potencia_14inf = 0
                        for i in range(len(fftfreq)):
                            if abs(fftfreq[i]) <= 2:
                                potencia_02 = potencia_02 + abs(dato_fft[i][0])**2
                            elif abs(fftfreq[i]) < 8:
                                potencia_28 = potencia_28 + abs(dato_fft[i][0])**2
                            elif abs(fftfreq[i]) < 14:
                                potencia_814 = potencia_814 + abs(dato_fft[i][0])**2
                            else:
                                potencia_14inf = potencia_14inf + abs(dato_fft[i][0])**2
                        descriptores[5].append(potencia_02/2)
                        descriptores[6].append(potencia_28/6)
                        descriptores[7].append(potencia_814/6)
                        descriptores[8].append(potencia_14inf/(max(fftfreq)-14))
                    break
                    
            s_FirstInd = s_LastInd
    
#%%
despierto = 0
ligero = 0
profundo = 0
rem = 0
no_se_sabe = 0
etiquetas_reales = []
for etiqueta in etiquetas:
    if etiqueta == "Sleep stage W":
        etiquetas_reales.append(0)  # Depierto = 0
        despierto = despierto + 1
    elif etiqueta == "Sleep stage 1" or etiqueta == "Sleep stage 2":
        etiquetas_reales.append(1)  # Ligero = 1
        ligero = ligero + 1
    elif etiqueta == "Sleep stage 3" or etiqueta == "Sleep stage 4":
        etiquetas_reales.append(2)  # Profundo = 2
        profundo = profundo + 1
    elif etiqueta == "Sleep stage R":
        etiquetas_reales.append(3)  # REM = 3
        rem = rem + 1
    else:
        etiquetas_reales.append(4)  # NPI = 4
        no_se_sabe = no_se_sabe + 1

            
#%%
train_size = int(len(etiquetas_reales) * 0.8)
test_size = int(len(etiquetas_reales) * 0.1)

X_train = datos[0:train_size]
y_train = np.array([etiquetas_reales[0:train_size]]).transpose()
y_train = to_categorical(y_train, 5)
X_test = datos[train_size:train_size+test_size]
y_test = np.array([etiquetas_reales[train_size:train_size+test_size]]).transpose()
y_test = to_categorical(y_test, 5)
X_val = datos[train_size+test_size:len(etiquetas_reales)]
y_val = np.array([etiquetas_reales[train_size+test_size:len(etiquetas_reales)]]).transpose()
y_val = to_categorical(y_val, 5)

descriptores_np = np.array(descriptores).transpose()
minimo = descriptores_np.min(axis=0)
maximo = descriptores_np.max(axis=0)
#media = descriptores_np.mean(axis=0)
#desv = descriptores_np.std(axis=0)
descriptores_norm = ((descriptores_np - minimo)/(maximo-minimo))

descriptores_train = None
descriptores_test = None
descriptores_val = None
for i in range(len(descriptores_norm)):
    descriptores_train = descriptores_norm[0:train_size, :]
    descriptores_test = descriptores_norm[train_size:train_size+test_size, :]
    descriptores_val = descriptores_norm[train_size+test_size:len(etiquetas_reales), :]

#%%
# Creación de modelos a probar
modelos = []
funcion_activacion = 'relu'
initializer = tf.keras.initializers.glorot_uniform(seed=0)
# Creación de la red
modelo = tf.keras.Sequential()
modelo.add(layers.Dense(750, input_shape = (descriptores_np.shape[1],),
                  activation = funcion_activacion))
modelo.add(Dropout(rate=0.5))
modelo.add(layers.Dense(500, 
                  activation = funcion_activacion))
modelo.add(Dropout(rate=0.5))
modelo.add(layers.Dense(250, 
                  activation = funcion_activacion))
modelo.add(Dropout(rate=0.5))
modelo.add(layers.Dense(50, 
                  activation = funcion_activacion))
modelo.add(Dropout(rate=0.5))
modelo.add(layers.Dense(5, activation = "softmax"))
Optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
Loss = tf.keras.losses.CategoricalCrossentropy()
modelo.compile(optimizer=Optimizer,
               loss=Loss,
               metrics=['accuracy'])
history = modelo.fit(x=descriptores_train,
                     y=y_train,
                     steps_per_epoch=20,
                     epochs=30,
                     validation_data=(descriptores_val, y_val),
                     initial_epoch=0)

#%%
prediccion_test = modelo.predict(x=descriptores_test)

#%%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

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
