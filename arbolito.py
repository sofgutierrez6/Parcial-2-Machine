# %%
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib 
from keras.utils import to_categorical
import pyedflib as pyedf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import scipy as sp
import scipy.fftpack
warnings.filterwarnings('ignore')
#%%
etiquetasTotales_np = np.load('./etiquetas.npy')
descriptoresTotales_np = np.load('./descriptores.npy')
#%%
etiquetas_cat = to_categorical(etiquetasTotales_np, 4)
descriptores_np = descriptoresTotales_np.transpose()
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
print('Entrenando modelo Random Forest')
model_rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=10,random_state=0, n_jobs=2)
model_rf.fit(des_train, etiq_train)
#%% Realizar prediccion en datos de testeo
print('Prediccion en datos de testo:')
etiq_pred = model_rf.predict(des_test)
precision = accuracy_score(etiq_test, etiq_pred)
print("%.4f" %precision)
#Realizar prediccion en datos de validación
print('Prediccion en datos de validación:')
etiq_pred_val = model_rf.predict(des_val)
precision_val = accuracy_score(etiq_val, etiq_pred_val)
print("%.4f" %precision_val)
#%% Matriz de confusion
print('Matriz de confusión')
y_descatergorizado_test = np.argmax(etiq_test, axis=1)
y_descatergorizado_pred = np.argmax(etiq_pred, axis=1)
tabla=pd.crosstab(y_descatergorizado_test, y_descatergorizado_pred, rownames=['Actual'], colnames=['Predicción'])
print(tabla*100/len(etiq_pred))
#%%
print(tabla)
#%% Reporte de classificación 
import sklearn
print(sklearn.metrics.classification_report(y_descatergorizado_test, y_descatergorizado_pred))
#%% Encontrar importancia de cada variable, y graficar
print('Calculando importancia de variables para prediccion')
importanciaVars=model_rf.feature_importances_
#%% Graficar con barras la importancia de cada variable
pos=range(27)
plt.rcdefaults()
fig, ax = plt.subplots()
ax.barh(pos, importanciaVars, align='center',color='blue')
ax.set_yticks(pos)
ax.set_yticklabels(pos)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Importancia Variables')
plt.show()
#%%
joblib.dump(model_rf, 'RandomForest3Canales.pkl')
#%%
print('Entrenando modelo Random Forest 2')
model_rf2 = RandomForestClassifier(n_estimators=100, max_features=4, min_samples_leaf=10,random_state=0, n_jobs=2)
model_rf2.fit(des_train, etiq_train)
print('Prediccion en datos de testo:')
etiq_pred2 = model_rf2.predict(des_test)
precision2 = accuracy_score(etiq_test, etiq_pred2)
print("%.4f" %precision)
#Realizar prediccion en datos de validación
print('Prediccion en datos de validación:')
etiq_pred_val2 = model_rf2.predict(des_val)
precision_val2 = accuracy_score(etiq_val, etiq_pred_val2)
print("%.4f" %precision_val2)