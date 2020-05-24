# -*- coding: utf-8 -*-
"""
Created on Sat May 23 16:51:41 2020

@author: Sony
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.utils import to_categorical
from keras import layers
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
# Lectura del archivo de estados de sueño (etiquetas)
st_FileHypEdf_Petit = pyedf.EdfReader("sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4001EC-Hypnogram.edf")
print(st_FileHypEdf_Petit)

# Datos en ventanas de 30 segundos, 
#v_HypTime es el tiempo de inicio, v_HypDur es la duración en un estado específico (pueden ser varias ventanas),
# v_Hyp es la etiqueta.

v_HypTime_Petit, v_HypDur_Petit, v_Hyp_Petit = st_FileHypEdf_Petit.readAnnotations()

# Lectura de las señales s_SigNum señales con nombres v_Signal_Labels
st_FileEdf_Petit = pyedf.EdfReader("sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4001E0-PSG.edf")
print(st_FileEdf_Petit)
s_SigNum_Petit = st_FileEdf_Petit.signals_in_file
print(s_SigNum_Petit)
v_Signal_Labels_Petit = st_FileEdf_Petit.getSignalLabels()

# Conversion a segundos usando frecuencia de muestreo.
s_SigRef_Petit = 0
s_NSamples_Petit = st_FileEdf_Petit.getNSamples()[0]
s_FsHz_Petit = st_FileEdf_Petit.getSampleFrequency(s_SigRef_Petit)
print(s_NSamples_Petit)

# v_Sig = np.zeros((s_NSamples, 1))
v_Sig_Petit = st_FileEdf_Petit.readSignal(s_SigRef_Petit)
v_Time_Petit = np.arange(0, s_NSamples_Petit) / s_FsHz_Petit

s_WinSizeSec_Petit = 30
s_WinSizeSam_Petit = np.round(s_FsHz_Petit * s_WinSizeSec_Petit)

#%%
datos = []
etiquetas = []
descriptores = [[] for _ in range(9)]
arreglo_x = np.array(list(range(3000)))
s_FirstInd_Petit = 0
while 1:
    s_LastInd_Petit = s_FirstInd_Petit + s_WinSizeSam_Petit
    if s_LastInd_Petit > s_NSamples_Petit:
        break

    for i in range(len(v_Hyp_Petit)-1):
        if ((v_HypTime_Petit[i] <= s_FirstInd_Petit/s_FsHz_Petit) and (s_FirstInd_Petit/s_FsHz_Petit < v_HypTime_Petit[i+1])):
            if v_Hyp_Petit[i] == "Sleep stage W" or v_Hyp_Petit[i] == "Sleep stage 1" or v_Hyp_Petit[i] == "Sleep stage 2" or v_Hyp_Petit[i] == "Sleep stage 3" or v_Hyp_Petit[i] == "Sleep stage 4" or v_Hyp_Petit[i] == "Sleep stage R":
                etiquetas.append(v_Hyp_Petit[i])
                datos.append(np.array(v_Sig_Petit[s_FirstInd_Petit:s_LastInd_Petit]))
                dato_df = pd.DataFrame(v_Sig_Petit[s_FirstInd_Petit:s_LastInd_Petit])
                descriptores[9*s_SigRef_Petit].append(dato_df.mean()[0])
                descriptores[9*s_SigRef_Petit + 1].append(dato_df.var()[0])
                descriptores[9*s_SigRef_Petit + 2].append(dato_df.kurt()[0])
                descriptores[9*s_SigRef_Petit + 3].append(dato_df.skew()[0])
                descriptores[9*s_SigRef_Petit + 4].append(len(pyaC.zerocross1d(x=arreglo_x, y=np.array(v_Sig_Petit[s_FirstInd_Petit:s_LastInd_Petit]))))
                # Frecuencia
                dato_fft = sp.fftpack.fft(dato_df)
                fftfreq = sp.fftpack.fftfreq(len(dato_fft), 1/s_FsHz_Petit)
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
                descriptores[9*s_SigRef_Petit + 5].append(potencia_02/2)
                descriptores[9*s_SigRef_Petit + 6].append(potencia_28/6)
                descriptores[9*s_SigRef_Petit + 7].append(potencia_814/6)
                descriptores[9*s_SigRef_Petit + 8].append(potencia_14inf/(max(fftfreq)-14))
            break
    s_FirstInd_Petit  = s_LastInd_Petit

#%%
from sklearn.model_selection import train_test_split
etiquetas_np = np.array(etiquetas) 
X_datos_1, X_datos_2 , y_datos_1, y_datos_2 = train_test_split(datos, etiquetas_np, test_size=0.5, random_state=4)   
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
        no_se_sabe = no_se_sabe + 1
y_np = np.array(etiquetas_reales)
#%%
X_datos = np.concatenate((X_datos_1, X_datos_2))
y_datos = np.concatenate((y_datos_1, y_datos_2))
#%%
y = to_categorical(y_np, 4)
descriptores_np = np.array(descriptores).transpose()
minimo = descriptores_np.min(axis=0)
maximo = descriptores_np.max(axis=0)
#media = descriptores_np.mean(axis=0)
#desv = descriptores_np.std(axis=0)
descriptores_norm = ((descriptores_np - minimo)/(maximo-minimo))

X_train, X_prueba, y_train_1, y_prueba_1 = train_test_split(datos, y, test_size=0.2, random_state=4)
X_test, X_val, y_test_1, y_val_1 = train_test_split(X_prueba, y_prueba_1, test_size=0.5, random_state=4)

des_train, des_prueba, y_train_2, y_prueba_2 = train_test_split(descriptores_norm, y, test_size=0.2, random_state=4)
des_test, des_val, y_test_2, y_val_2 = train_test_split(des_prueba, y_prueba_2, test_size=0.5, random_state=4)

#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print('Entrenando modelo Random Forest ...')
model_rf = RandomForestClassifier(n_estimators=100, max_features=4, min_samples_leaf=10,random_state=0, n_jobs=2)
model_rf.fit(des_train, y_train_2)
#%%
# Encontrar importancia de cada variable, y graficar
print('Calculando importancia de variables para prediccion ...')
importanciaVars=model_rf.feature_importances_
# Graficar con barras la importancia de cada variable
pos=[1, 2, 3, 4, 5, 6, 7, 8, 9]
plt.rcdefaults()
fig, ax = plt.subplots()
ax.barh(pos, importanciaVars, align='center',color='blue')
ax.set_yticks(pos)
ax.set_yticklabels(pos)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Importancia Variables')
plt.show()
#%% Realizar prediccion en datos de validacion
print(""" Prediccion en datos de validacion...""")
y_pred = model_rf.predict(des_prueba)
precision=accuracy_score(y_prueba_2, y_pred)
print("%.4f" %precision)

#%% Matriz de confusion
print(""" Graficar matriz de confusión...""")
y_descatergorizado_test = np.argmax(y_prueba_2, axis=1)
y_descatergorizado_pred = np.argmax(y_pred, axis=1)
tabla=pd.crosstab(y_descatergorizado_test, y_descatergorizado_pred, rownames=['Actual'], colnames=['Predicción'])
print(tabla*100/len(y_pred))
#%%
print(tabla)
#%%
#classification Report
import sklearn
print(sklearn.metrics.classification_report(y_descatergorizado_test, y_descatergorizado_pred))
#%%
from sklearn.externals import joblib 
joblib.dump(model_rf, 'RandomForest.pkl')