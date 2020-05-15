
import pyedflib as pyedf
import numpy as np
import matplotlib.pyplot as ptl


# Lectura del archivo de estados de sueño (etiquetas)
st_FileHypEdf = pyedf.EdfReader("./data/SC4001EC-Hypnogram.edf")
print(st_FileHypEdf)

# Datos en ventanas de 30 segundos, 
#v_HypTime es el tiempo de inicio, v_HypDur es la duración en un estado específico (pueden ser varias ventanas),
# v_Hyp es la etiqueta.

v_HypTime, v_HypDur, v_Hyp = st_FileHypEdf.readAnnotations()

ptl.figure()
ptl.plot(v_HypTime, v_Hyp)


# Lectura de las señales s_SigNum señales con nombres v_Signal_Labels
st_FileEdf = pyedf.EdfReader("./data/SC4001E0-PSG.edf")
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
#ptl.figure()
while 1:
    s_LastInd = s_FirstInd + s_WinSizeSam
    if s_LastInd > s_NSamples:
        break
    datos.append(v_Sig[s_FirstInd:s_LastInd])
    for i in range(len(v_Hyp) - 1):
        if ((v_HypTime[i] <= s_FirstInd/s_FsHz) and (s_FirstInd/s_FsHz < v_HypTime[i+1])):
            etiquetas.append(v_Hyp[i])
            break
    s_FirstInd = s_LastInd
    
#%%
despierto = 0
ligero = 0
profundo = 0
rem = 0
etiquetas_reales = []
for etiqueta in etiquetas:
    if etiqueta == "Sleep stage W":
        etiquetas_reales.append("Despierto")
        despierto = despierto + 1
    elif etiqueta == "Sleep stage 1" or etiqueta == "Sleep stage 2":
        etiquetas_reales.append("Ligero")
        ligero = ligero + 1
    elif etiqueta == "Sleep stage 3" or etiqueta == "Sleep stage 4":
        etiquetas_reales.append("Profundo")
        profundo = profundo + 1
    elif etiqueta == "Sleep stage R":
        etiquetas_reales.append("REM")
        rem = rem + 1
    
#ptl.show()

'''
ptl.plot(v_Time[s_FirstInd:s_LastInd], v_Sig[s_FirstInd:s_LastInd], linewidth=1)
ptl.xlabel('Time (sec)')
ptl.xlim(v_Time[s_FirstInd], v_Time[s_LastInd - 1])
ptl.pause(0.05) '''

#%%
import pandas as pd

datos_df = pd.DataFrame(data=datos)
kurtosis = datos_df.kurt(axis=1)
promedio = datos_df.mean(axis=1)
varianza = datos_df.var(axis=1)
oblicuidad = datos_df.skew(axis=1)
#%% 
from entropy import sample_entropy
entropia = []
for i in range(datos_df.shape[0]):
    entropia.append(sample_entropy(datos_df.iloc[i,:], order=2, metric='chebyshev'))  
entropia = pd.DataFrame(entropia)

#%%
from PyAstronomy import pyaC
cruces_por_cero = []
arreglo_x = np.array(list(range(datos_df.shape[1])))
for i in range(datos_df.shape[0]):
    cruces_por_cero.append(len(pyaC.zerocross1d(x=arreglo_x, y=np.array(list(datos_df.iloc[i,:])))))
cruces_por_cero = pd.DataFrame(cruces_por_cero)

#%%
import yasa
#yasa.spindles_detect(data, sf, remove_outliers=True)  # Spindles
slow_waves = []
for i in range(datos_df.shape[0]):
    slow_waves.append(yasa.sw_detect(datos_df.iloc[i,:], s_FsHz, remove_outliers=True))
slow_waves = pd.DataFrame(slow_waves)
#yasa.rem_detect(loc, roc, sf, remove_outliers=True)   # REMs
