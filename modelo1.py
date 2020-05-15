import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#####################################################
# Cargar datos
print('Cargando datos...')
X = pd.read_csv('DataTrain.csv', header = None)
X = X.values[:,0:27]

y= pd.read_csv('LabelsTrain.csv', header = None)

feat_labels = ['Numero de Puntos','Radio','Ancho','DesciEstCentro','RelacionAspecto','LonFronteras','RegFronteras','Circularidad','Curtosis', 'MedAngular','DesPromMed','CurMedi','DistClos','RadioVecinoCerca','AnchoVecinoCerca','DesEstVeCer','AspectoVecCercca','LongFronVecinCer','RegulFronteVecino','CircuVecCercano','curtosisVecinoCercano','linearidadVeCERCANO','DifMedAnVeCe','DesProMedianaVecCe','CurvMediaVecCera','distaVeVeCer','JDFVBAJDF']
#print('labels cantidad',len(feat_labels))
#print('x', X)
#print('xShape', X.shape)
#print('y', y)


# Separar en datos de entrenamiento y validacion
# (Xtrain,y_train): datos de entrenamiento
# (Xtest,y_test): datos de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# print('X_train', X_train.shape)
# print('X_test', X_test.shape)
# print('y_train', y_train.shape)
# print('y_test', y_test.shape)

######################################################
# Entrenar clasificador
print('Entrenando modelo Random Forest ...')

model_rf = RandomForestClassifier(n_estimators=100, max_features=8, min_samples_leaf=10,random_state=0, n_jobs=2)
model_rf.fit(X_train, y_train.values.ravel())



#####################################################
# Encontrar importancia de cada variable, y graficar
print('Calculando importancia de variables para prediccion ...')
importanciaVars=model_rf.feature_importances_

print('importanciaVars',importanciaVars)

# Graficar con barras la importancia de cada variable
pos= range(0,27)
plt.rcdefaults()
fig, ax = plt.subplots()
ax.barh(pos, importanciaVars, align='center',color='blue')
ax.set_yticks(pos)
ax.set_yticklabels(feat_labels)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Importancia Variables')
plt.show()


######################################################
# Realizar prediccion en datos de validacion
print("""
      Prediccion en datos de validacion...""")
y_pred = model_rf.predict(X_test)
precision=accuracy_score(y_test, y_pred)
print("%.4f" %precision)

# Matriz de confusion
print("""
      Graficar matriz de confusión...
      """)
tabla=pd.crosstab(y_test.values.ravel(), y_pred, rownames=['Actual LOS'], colnames=['Predicted LOS'])
print(tabla*100/len(y_pred))






