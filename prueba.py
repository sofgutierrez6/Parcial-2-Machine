#%%
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
import matplotlib.pyplot as ptl
import pandas as pd
from PyAstronomy import pyaC
import warnings
import scipy as sp
import scipy.fftpack
warnings.filterwarnings('ignore')
#%%
# kurtosis = datos_df.kurt(axis=1)
# promedio = datos_df.mean(axis=1)
# varianza = datos_df.var(axis=1)
# oblicuidad = datos_df.skew(axis=1)
#%% 
# entropia = []
# for i in range(datos_df.shape[0]):
#     entropia.append(sample_entropy(datos_df.iloc[i,:], order=2, metric='chebyshev'))  
# entropia = pd.DataFrame(entropia)

#%%
# cruces_por_cero = []
# for i in range(datos_df.shape[0]):
#     cruces_por_cero.append(len(pyaC.zerocross1d(x=arreglo_x, y=np.array(list(datos_df.iloc[i,:])))))
# cruces_por_cero = pd.DataFrame(cruces_por_cero)

#%%
#yasa.spindles_detect(data, sf, remove_outliers=True)  # Spindles
# slow_waves = []
# for i in range(datos_df.shape[0]):
#     slow_waves.append(yasa.sw_detect(datos_df.iloc[i,:], s_FsHz, remove_outliers=True))
# slow_waves = pd.DataFrame(slow_waves)
#yasa.rem_detect(loc, roc, sf, remove_outliers=True)   # REMs

#%%

potencia_0_2 = []
potencia_2_8 = []
potencia_8_14 = []
potencia_14_inf = []

for dato in range(datos_df.shape[0]):
    dato_fft = sp.fftpack.fft(dato)
    fftfreq = sp.fftpack.fftfreq(len(dato_fft), s_FsHz)
    potencia_02 = 0
    potencia_28 = 0
    potencia_814 =  0
    potencia_14inf = 0
    for i in range(len(fftfreq)):
        if fftfreq[i] <= 2:
            potencia_02 = potencia_02 + abs(dato_fft[i])**2
        elif fftfreq[i] > 2 and fftfreq[i] < 8:
            potencia_28 = potencia_28 + abs(dato_fft[i])**2
        elif fftfreq[i] >= 8 and fftfreq[i] < 14:
            potencia_814 = potencia_814 + abs(dato_fft[i])**2
        else:
            potencia_14inf = potencia_14inf + abs(dato_fft[i])**2
#%%
# Lectura del archivo de estados de sueño (etiquetas)
st_FileHypEdf = pyedf.EdfReader(ruta_archivos +"/SC4011EH-Hypnogram.edf")
print(st_FileHypEdf)

# Datos en ventanas de 30 segundos, 
#v_HypTime es el tiempo de inicio, v_HypDur es la duración en un estado específico (pueden ser varias ventanas),
# v_Hyp es la etiqueta.

v_HypTime, v_HypDur, v_Hyp = st_FileHypEdf.readAnnotations()

ptl.figure()
ptl.plot(v_HypTime, v_Hyp)


# Lectura de las señales s_SigNum señales con nombres v_Signal_Labels
st_FileEdf = pyedf.EdfReader(ruta_archivos + "/SC4011E0-PSG.edf")
print(st_FileEdf)
s_SigNum = st_FileEdf.signals_in_file
print(s_SigNum)
v_Signal_Labels = st_FileEdf.getSignalLabels()


# Conversion a segundos usando frecuencia de muestreo.
s_SigRef = 0
s_NSamples = st_FileEdf.getNSamples()[0]
s_FsHz = st_FileEdf.getSampleFrequency(s_SigRef)
print(s_NSamples)

# v_Sig = np.zeros((s_NSamples, 1))
v_Sig = st_FileEdf.readSignal(s_SigRef)
v_Time = np.arange(0, s_NSamples) / s_FsHz

s_WinSizeSec = 5
s_WinSizeSam = np.round(s_FsHz * s_WinSizeSec)

#%%
# plot de señales en ventanas de 30s
s_FirstInd = 0
datos = []
etiquetas = []
descriptores = [[] for _ in range(11)]
arreglo_x = np.array(list(range(500)))
#ptl.figure()
while 1:
    s_LastInd = s_FirstInd + s_WinSizeSam
    if s_LastInd > 501:
        break
    for i in range(len(v_Hyp)-1):
        if ((v_HypTime[i] <= s_FirstInd/s_FsHz) and (s_FirstInd/s_FsHz < v_HypTime[i+1])):
            if v_Hyp[i] != "Sleep stage ?":
                etiquetas.append(v_Hyp[i])
                datos.append(v_Sig[s_FirstInd:s_LastInd])
                dato_df = pd.DataFrame(v_Sig[s_FirstInd:s_LastInd])
                descriptores[0].append(dato_df.mean()[0])
                descriptores[1].append(dato_df.var()[0])
                descriptores[2].append(dato_df.kurt()[0])
                descriptores[3].append(dato_df.skew()[0])
                descriptores[4].append(len(pyaC.zerocross1d(x=arreglo_x, y=np.array(v_Sig[s_FirstInd:s_LastInd]))))
                sw = yasa.sw_detect(dato_df.transpose(), s_FsHz, remove_outliers=True)
                descriptores[5].append(sw.summary().shape[0])
                suma = 0
                for i in range(sw.summary().shape[0]):
                    suma = suma + (sw.summary()['Duration'].iloc[i])
                descriptores[6].append(suma)
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
                descriptores[7].append(potencia_02/2)
                descriptores[8].append(potencia_28/6)
                descriptores[9].append(potencia_814/6)
                descriptores[10].append(potencia_14inf/(max(fftfreq)-14))
            break            
    s_FirstInd = s_LastInd
    

#%%
ptl.plot(datos[0], linewidth=1)
ptl.xlabel('Time (sec)')
ptl.pause(0.05)
print(etiquetas[0])