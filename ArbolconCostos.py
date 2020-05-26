etiquetasTotales_np = np.load('./etiquetas.npy')
descriptoresTotales_np = np.load('./descriptores.npy')
#%%
descriptores_np = descriptoresTotales_np.transpose()
minimo = descriptores_np.min(axis=0)
maximo = descriptores_np.max(axis=0)
#media = descriptores_np.mean(axis=0)
#desv = descriptores_np.std(axis=0)
descriptores_norm = ((descriptores_np - minimo)/(maximo-minimo))

#dat_train, dat_prueba, etiq_train_1, etiq_prueba_1 = train_test_split(datos, etiquetas_np, test_size=0.2, random_state=4)
#dat_test, dat_val, y_test_1, y_val_1 = train_test_split(X_prueba, y_prueba_1, test_size=0.5, random_state=4)

des_train, des_prueba, etiq_train, etiq_prueba = train_test_split(descriptores_norm, descriptoresTotales_np, test_size=0.2, random_state=4)
des_test, des_val, etiq_test, etiq_val = train_test_split(des_prueba, etiq_prueba, test_size=0.5, random_state=4)
#%%
des_train, des_prueba, etiq_train, etiq_prueba = train_test_split(descriptores_norm, etiquetasTotales_np, test_size=0.2, random_state=4)
des_test, des_val, etiq_test, etiq_val = train_test_split(des_prueba, etiq_prueba, test_size=0.5, random_state=4)

weights = dict([{0:1, 1:1}, {0:1, 1:1}, {0:1, 1:100} , {0:1, 1:1}])
model_weight = RandomForestClassifier(class_weight=weights, n_estimators=100, min_samples_leaf=10,random_state=0, n_jobs=2)
model_weight.fit(des_train, etiq_train)
#%%
print('Prediccion en datos de testo:')
etiq_pred_weight = model_weight.predict(des_test)
precision2_weight = accuracy_score(etiq_test, etiq_pred_mejor)
print("%.4f" %precision2_weight)
#Realizar prediccion en datos de validación
print('Prediccion en datos de validación:')
etiq_pred_val_weight = model_weight.predict(des_val)
precision_val2_weight = accuracy_score(etiq_val, etiq_pred_val_weight)
print("%.4f" %precision_val2_weight)
#%%
import sklearn
print(sklearn.metrics.classification_report(etiq_test, etiq_pred_weight))
#%% Encontrar importancia de cada variable, y graficar
import matplotlib.pyplot as plt
print('Calculando importancia de variables para prediccion')
importanciaVars=model_weight.feature_importances_
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
tabla=pd.crosstab(etiq_test, etiq_pred_weight, rownames=['Actual'], colnames=['Predicción'])
print(tabla*100/len(etiq_pred_mejor))
print(tabla)
#%%
##########################
#
'''Validación cruzada'''
#
#########################
from sklearn.model_selection import cross_validate
model_crossweight = RandomForestClassifier(class_weight=weights, random_state = 1)
cv = cross_validate(model_crossweight, descriptores_norm, etiquetasTotales_np, cv = 6, return_estimator = True)
print(cv['test_score'])
print(cv['test_score'].mean())
#%%
mejor_weight = cv['estimator'][0]
print('Prediccion en datos de testo:')
etiq_pred_mejor_weight = mejor_weight.predict(des_test)
precision2_mejor_weight = accuracy_score(etiq_test, etiq_pred_mejor_weight)
print("%.4f" %precision2_mejor_weight)
#Realizar prediccion en datos de validación
print('Prediccion en datos de validación:')
etiq_pred_val_mejor_weight = mejor_weight.predict(des_val)
precision_val2_mejor_weight = accuracy_score(etiq_val, etiq_pred_val_mejor_weight)
print("%.4f" %precision_val2_mejor_weight)
#%%
import sklearn
print(sklearn.metrics.classification_report(etiq_test, etiq_pred_mejor_weight))
#%% Encontrar importancia de cada variable, y graficar
import matplotlib.pyplot as plt
print('Calculando importancia de variables para prediccion')
importanciaVars=mejor_weight.feature_importances_
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
tabla=pd.crosstab(etiq_test, etiq_pred_mejor_weight, rownames=['Actual'], colnames=['Predicción'])
print(tabla*100/len(etiq_pred_mejor_weight))
print(tabla)

joblib.dump(model_weight, 'RandomForest3CanalesMejoradoPesado.pkl')
