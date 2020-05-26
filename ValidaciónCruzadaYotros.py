from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
model_rf = joblib.load('RandomForest.pkl')
model_rf.fit(des_train[:,0:9], etiq_train)
#%%
print('Prediccion en datos de testo:')
etiq_pred2 = model_rf.predict(des_test[:,0:9])
precision2 = accuracy_score(etiq_test, etiq_pred2)
print("%.4f" %precision2)
#Realizar prediccion en datos de validación
print('Prediccion en datos de validación:')
etiq_pred_val2 = model_rf.predict(des_val[:,0:9])
precision_val2 = accuracy_score(etiq_val, etiq_pred_val2)
print("%.4f" %precision_val2)
#%%
from sklearn.model_selection import cross_validate

etiquetasTotales_np = np.load('./etiquetas.npy')
descriptoresTotales_np = np.load('./descriptores.npy')

descriptores_np = descriptoresTotales_np.transpose()
minimo = descriptores_np.min(axis=0)
maximo = descriptores_np.max(axis=0)
#media = descriptores_np.mean(axis=0)
#desv = descriptores_np.std(axis=0)
descriptores_norm = ((descriptores_np - minimo)/(maximo-minimo))

#dat_train, dat_prueba, etiq_train_1, etiq_prueba_1 = train_test_split(datos, etiquetas_np, test_size=0.2, random_state=4)
#dat_test, dat_val, y_test_1, y_val_1 = train_test_split(X_prueba, y_prueba_1, test_size=0.5, random_state=4)

des_train, des_prueba, etiq_train, etiq_prueba = train_test_split(descriptores_norm, etiquetasTotales_np, test_size=0.2, random_state=4)
des_test, des_val, etiq_test, etiq_val = train_test_split(des_prueba, etiq_prueba, test_size=0.5, random_state=4)
#%%
model_cross = RandomForestClassifier(random_state = 1)
cv = cross_validate(model_cross, descriptores_norm, etiquetasTotales_np, cv = 5, return_estimator = True)
print(cv['test_score'])
print(cv['test_score'].mean())
#%%
mejor = cv['estimator'][0]
#%% 
'''Se guardaaa'''
joblib.dump(mejor, 'RandomForest3CanalesMejorado.pkl')
#%%
'''Se importa'''
from sklearn.externals import joblib 
mejor = joblib.load('RandomForest3CanalesMejorado.pkl')

#%%
import numpy as np
etiquetasTotales_np = np.load('./etiquetas.npy')
descriptoresTotales_np = np.load('./descriptores.npy')
#%%
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
descriptores_np = descriptoresTotales_np.transpose()
minimo = descriptores_np.min(axis=0)
maximo = descriptores_np.max(axis=0)
#media = descriptores_np.mean(axis=0)
#desv = descriptores_np.std(axis=0)
descriptores_norm = ((descriptores_np - minimo)/(maximo-minimo))

#dat_train, dat_prueba, etiq_train_1, etiq_prueba_1 = train_test_split(datos, etiquetas_np, test_size=0.2, random_state=4)
#dat_test, dat_val, y_test_1, y_val_1 = train_test_split(X_prueba, y_prueba_1, test_size=0.5, random_state=4)

des_train, des_prueba, etiq_train, etiq_prueba = train_test_split(descriptores_norm, etiquetasTotales_np, test_size=0.2, random_state=4)
des_test, des_val, etiq_test, etiq_val = train_test_split(des_prueba, etiq_prueba, test_size=0.5, random_state=4)
#%%
print('Prediccion en datos de testo:')
etiq_pred_mejor = mejor.predict(des_test)
precision2_mejor = accuracy_score(etiq_test, etiq_pred_mejor)
print("%.4f" %precision2_mejor)
#Realizar prediccion en datos de validación
print('Prediccion en datos de validación:')
etiq_pred_val_mejor = mejor.predict(des_val)
precision_val2_mejor = accuracy_score(etiq_val, etiq_pred_val_mejor)
print("%.4f" %precision_val2_mejor)
#%%
import sklearn
print(sklearn.metrics.classification_report(etiq_test, etiq_pred_mejor))
#%% Encontrar importancia de cada variable, y graficar
import matplotlib.pyplot as plt
print('Calculando importancia de variables para prediccion')
importanciaVars=mejor.feature_importances_
#Graficar con barras la importancia de cada variable
pos=range(1,28)
plt.rcdefaults()
fig, ax = plt.subplots()
ax.barh(pos, importanciaVars, align='center',color='blue')
ax.set_yticks(pos)
ax.set_yticklabels(pos)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Importancia Variables')
plt.show()
#%%
import pandas as pd
tabla=pd.crosstab(etiq_test, etiq_pred_mejor, rownames=['Actual'], colnames=['Predicción'])
print(tabla*100/len(etiq_pred_mejor))
print(tabla)
