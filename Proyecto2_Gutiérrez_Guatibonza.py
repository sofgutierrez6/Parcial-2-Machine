# %%
from os import scandir
import pyedflib as pyedf
import numpy as np
import matplotlib.pyplot as ptl
import pandas as pd
from entropy import sample_entropy
from PyAstronomy import pyaC
import yasa
import numpy as np
import scipy as sp
import scipy.fftpack
import pandas as pd

# %%
ruta_data = "../../data/sleep-edf-database-expanded-1.0.0/sleep-cassette"
lista_archivos = [arch.name for arch in scandir(ruta_data) if arch.is_file()]
# %%
datos = []
etiquetas = []
contador = 0
for ruta_actual in lista_archivos:
    if ruta_actual[9:len(ruta_actual)] == "Hypnogram.edf":
        identificador = ruta_actual[2:7]
        
        # Lectura del archivo de estados de sueño (etiquetas)
        st_FileHypEdf = pyedf.EdfReader(ruta_data + "/" + ruta_actual)
        # Datos en ventanas de 30 segundos, 
        # v_HypTime es el tiempo de inicio, v_HypDur es la duración en un estado específico (pueden ser varias ventanas),
        # v_Hyp es la etiqueta.
        v_HypTime, v_HypDur, v_Hyp = st_FileHypEdf.readAnnotations()
        
        # Lectura de las señales s_SigNum señales con nombres v_Signal_Labels
        st_FileEdf = pyedf.EdfReader(ruta_data + "/SC" + identificador + "0-PSG.edf")
        s_SigNum = st_FileEdf.signals_in_file
        v_Signal_Labels = st_FileEdf.getSignalLabels()
        
        # Conversion a segundos usando frecuencia de muestreo.
        s_SigRef = 0
        s_NSamples = st_FileEdf.getNSamples()[0]
        s_FsHz = st_FileEdf.getSampleFrequency(s_SigRef)
        
        # v_Sig = np.zeros((s_NSamples, 1))
        v_Sig = st_FileEdf.readSignal(s_SigRef)
        v_Time = np.arange(0, s_NSamples) / s_FsHz
        
        s_WinSizeSec = 5
        s_WinSizeSam = np.round(s_FsHz * s_WinSizeSec)
        s_FirstInd = 0
        while 1:
            s_LastInd = s_FirstInd + s_WinSizeSam
            if s_LastInd > s_NSamples:
                break
            for i in range(len(v_Hyp)):
                if i != len(v_Hyp)-1:
                    if ((v_HypTime[i] <= s_FirstInd/s_FsHz) and (s_FirstInd/s_FsHz < v_HypTime[i+1])):
                        etiquetas.append(v_Hyp[i])
                        datos.append(v_Sig[s_FirstInd:s_LastInd])
                        break
                else:
                    if (s_FirstInd/s_FsHz > v_HypTime[i]):
                        etiquetas.append(v_Hyp[i])
                        datos.append(v_Sig[s_FirstInd:s_LastInd])
                    
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
    else:
        etiquetas_reales.append("NPI")
        no_se_sabe = no_se_sabe + 1
    
#ptl.show()

'''
ptl.plot(v_Time[s_FirstInd:s_LastInd], v_Sig[s_FirstInd:s_LastInd], linewidth=1)
ptl.xlabel('Time (sec)')
ptl.xlim(v_Time[s_FirstInd], v_Time[s_LastInd - 1])
ptl.pause(0.05) '''

#%%

datos_df = pd.DataFrame(data=datos)
kurtosis = datos_df.kurt(axis=1)
promedio = datos_df.mean(axis=1)
varianza = datos_df.var(axis=1)
oblicuidad = datos_df.skew(axis=1)
#%% 
entropia = []
for i in range(datos_df.shape[0]):
    entropia.append(sample_entropy(datos_df.iloc[i,:], order=2, metric='chebyshev'))  
entropia = pd.DataFrame(entropia)

#%%
cruces_por_cero = []
arreglo_x = np.array(list(range(datos_df.shape[1])))
for i in range(datos_df.shape[0]):
    cruces_por_cero.append(len(pyaC.zerocross1d(x=arreglo_x, y=np.array(list(datos_df.iloc[i,:])))))
cruces_por_cero = pd.DataFrame(cruces_por_cero)

#%%
#yasa.spindles_detect(data, sf, remove_outliers=True)  # Spindles
slow_waves = []
for i in range(datos_df.shape[0]):
    slow_waves.append(yasa.sw_detect(datos_df.iloc[i,:], s_FsHz, remove_outliers=True))
slow_waves = pd.DataFrame(slow_waves)
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
            potencia_02 = potencia_0_2 + abs(dato_fft[i])**2
        elif fftfreq[i] > 2 and fftfreq[i] < 8:
            potencia_28 = potencia_2_8 + abs(dato_fft[i])**2
        elif fftfreq[i] >= 8 and fftfreq[i] < 14:
            potencia_814 = potencia_814 + abs(dato_fft[i])**2
        else:
            potencia_14inf = potencia_14inf + abs(dato_fft[i])**2
            
#%%
y_datos = pd.DataFrame(etiquetas_reales)
train_size = int(len(etiquetas_reales) * 0.8)
test_size = int(len(etiquetas_reales) * 0.1)

X_train = datos_df[0:train_size]
y_train = y_datos[0:train_size]
X_test = datos_df[train_size:train_size+test_size]
y_test = y_datos[train_size:train_size+test_size]
X_val = datos_df[train_size+test_size:y_datos.shape[0]]
y_val = y_datos[train_size+test_size:y_datos.shape[0]]

#%%
ptl.plot(datos[0], linewidth=1)
ptl.xlabel('Time (sec)')
ptl.pause(0.05)
print(etiquetas[0])