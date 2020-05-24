# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.model_selection import train_test_split
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
# %%
ruta_archivos = "../../data/sleep-edf-database-expanded-1.0.0/sleep-cassette"
lista_archivos = [arch.name for arch in scandir(ruta_archivos) if arch.is_file()]
# %%
num_canales = 3
datos = []
etiquetas = []
descriptores = [[] for _ in range(27)]
arreglo_x = np.array(list(range(3000)))
for ruta_actual in lista_archivos:
    if ruta_actual[9:len(ruta_actual)] == "Hypnogram.edf":
        identificador = ruta_actual[2:7]
        st_FileHypEdf = pyedf.EdfReader(ruta_archivos + "/" + ruta_actual)
        v_HypTime, v_HypDur, v_Hyp = st_FileHypEdf.readAnnotations()# Lectura de las señales s_SigNum señales con nombres v_Signal_Labels
        st_FileEdf = pyedf.EdfReader(ruta_archivos + "/SC" + identificador + "0-PSG.edf")
        s_SigNum = st_FileEdf.signals_in_file
        v_Signal_Labels = st_FileEdf.getSignalLabels()        
        
        datos_actual = [[] for _ in range(num_canales)]
        etiquetas_actual = []
        for s_SigRef in range(num_canales):
            s_NSamples = st_FileEdf.getNSamples()[0]
            s_FsHz = st_FileEdf.getSampleFrequency(s_SigRef)
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
                        if v_Hyp[i] == "Sleep stage W" or v_Hyp[i] == "Sleep stage 1" or v_Hyp[i] == "Sleep stage 2" or v_Hyp[i] == "Sleep stage 3" or v_Hyp[i] == "Sleep stage 4" or v_Hyp[i] == "Sleep stage R":
                            if s_SigRef == 0:
                                etiquetas_actual.append(v_Hyp[i])
                            datos_actual[s_SigRef].append(np.array(v_Sig[s_FirstInd:s_LastInd]))       
                        break
                s_FirstInd = s_LastInd
        datos_actual = np.array(datos_actual)
        etiquetas_actual = np.array(etiquetas_actual)
        
        despierto = 0
        ligero = 0
        profundo = 0
        rem = 0
        for etiqueta in etiquetas_actual:        
            if etiqueta == "Sleep stage W":
                despierto = despierto + 1
            elif etiqueta == "Sleep stage 1" or etiqueta == "Sleep stage 2":
                ligero = ligero + 1
            elif etiqueta == "Sleep stage 3" or etiqueta == "Sleep stage 4":
                profundo = profundo + 1
            elif etiqueta == "Sleep stage R":
                rem = rem + 1
        min_labels = min([despierto, ligero, profundo, rem])
        
        datos_actual_canales = []
        for i in range(datos_actual.shape[1]):
            datos_actual_canales.append(np.array(datos_actual[:,i,:]))
        
        X_datos_1, X_datos_2, y_datos_1, y_datos_2 = train_test_split(datos_actual_canales, etiquetas_actual, test_size=0.5, random_state=4)
        datos_shuffle = np.concatenate((X_datos_1, X_datos_2))
        etiquetas_shuffle = np.concatenate((y_datos_1, y_datos_2))
        
        despierto = 0
        ligero = 0
        profundo = 0
        rem = 0
        for i in range(etiquetas_shuffle.shape[0]):  
            agregar_dato = False
            if etiquetas_shuffle[i] == "Sleep stage W":
                if despierto < min_labels:
                    etiquetas.append(0)
                    agregar_dato = True
                despierto = despierto + 1
            elif etiquetas_shuffle[i] == "Sleep stage 1" or etiquetas_shuffle[i] == "Sleep stage 2":
                if ligero < min_labels:
                    etiquetas.append(1)
                    agregar_dato = True
                ligero = ligero + 1
            elif etiquetas_shuffle[i] == "Sleep stage 3" or etiquetas_shuffle[i] == "Sleep stage 4":
                if profundo < min_labels:
                    etiquetas.append(2)
                    agregar_dato = True
                profundo = profundo + 1
            elif etiquetas_shuffle[i] == "Sleep stage R":
                if rem < min_labels:
                    etiquetas.append(3)
                    agregar_dato = True
                rem = rem + 1
            if agregar_dato:
                datos.append(datos_shuffle[i,:,:])
                for s_SigRef in range(num_canales):
                    dato_df = pd.DataFrame(datos_shuffle[i,s_SigRef,:])
                    descriptores[9*s_SigRef].append(dato_df.mean()[0])
                    descriptores[9*s_SigRef + 1].append(dato_df.var()[0])
                    descriptores[9*s_SigRef + 2].append(dato_df.kurt()[0])
                    descriptores[9*s_SigRef + 3].append(dato_df.skew()[0])
                    descriptores[9*s_SigRef + 4].append(len(pyaC.zerocross1d(x=arreglo_x, y=dato_df.values[:,0])))
                    # Frecuencia
                    dato_fft = sp.fftpack.fft(dato_df)
                    fftfreq = sp.fftpack.fftfreq(len(dato_fft), 1/s_FsHz)
                    potencia_02 = 0
                    potencia_28 = 0
                    potencia_814 =  0
                    potencia_14inf = 0
                    for j in range(len(fftfreq)):
                        if abs(fftfreq[j]) <= 2:
                            potencia_02 = potencia_02 + abs(dato_fft[j][0])**2
                        elif abs(fftfreq[j]) < 8:
                            potencia_28 = potencia_28 + abs(dato_fft[j][0])**2
                        elif abs(fftfreq[j]) < 14:
                            potencia_814 = potencia_814 + abs(dato_fft[j][0])**2
                        else:
                            potencia_14inf = potencia_14inf + abs(dato_fft[j][0])**2
                    descriptores[9*s_SigRef + 5].append(potencia_02/2)
                    descriptores[9*s_SigRef + 6].append(potencia_28/6)
                    descriptores[9*s_SigRef + 7].append(potencia_814/6)
                    descriptores[9*s_SigRef + 8].append(potencia_14inf/(max(fftfreq)-14))

#%%
despierto = 0
ligero = 0
profundo = 0
rem = 0
no_se_sabe = 0
for etiqueta in etiquetas:
    if etiqueta == 0:
        # Depierto = 0
        despierto = despierto + 1
    elif etiqueta == 1:
        # Ligero = 1
        ligero = ligero + 1
    elif etiqueta == 2:
        # Profundo = 2
        profundo = profundo + 1
    elif etiqueta == 3:
        # REM = 3
        rem = rem + 1
    else:
        no_se_sabe = no_se_sabe + 1
        
#%%
datos_np = np.array(datos)
np.save('./datos.npy', datos_np)
etiquetas_np = np.array(etiquetas)
np.save('./etiquetas.npy', etiquetas_np)
descriptores_np = np.array(descriptores)
np.save('./descriptores.npy', descriptores_np)
