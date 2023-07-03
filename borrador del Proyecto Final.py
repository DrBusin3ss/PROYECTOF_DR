import pandas as pd
import numpy as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


dataframe = pd.read_csv('C:/Users/Usuario/Desktop/proyecto final/customers.csv', encoding= 'latin-1') 
print(dataframe.head())

dataframe.head()

x = dataframe['Gender','Subscription Type']
y = dataframe['Age']


import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Leer los datos
df = pd.read_csv('datos.csv')

# Dividir los datos en características (X) y variable objetivo (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Definir el número de folds para la validación cruzada
k = 5

# Naive Bayes
nb_model = GaussianNB()
nb_scores = cross_val_score(nb_model, X, y, cv=k)

# LDA
lda_model = LinearDiscriminantAnalysis()
lda_scores = cross_val_score(lda_model, X, y, cv=k)

# Regresión logística
logreg_model = LogisticRegression()
logreg_scores = cross_val_score(logreg_model, X, y, cv=k)

# SVM
svm_model = SVC()
svm_scores = cross_val_score(svm_model, X, y, cv=k)

# Árboles de decisión
dt_model = DecisionTreeClassifier()
dt_scores = cross_val_score(dt_model, X, y, cv=k)

# Random Forest
rf_model = RandomForestClassifier()
rf_scores = cross_val_score(rf_model, X, y, cv=k)

# Análisis de discriminante lineal
lda_model = LinearDiscriminantAnalysis()
lda_scores = cross_val_score(lda_model, X, y, cv=k)

# Análisis de discriminante cuadrático
qda_model = QuadraticDiscriminantAnalysis()
qda_scores = cross_val_score(qda_model, X, y, cv=k)

# AdaBoost
adaboost_model = AdaBoostClassifier()
adaboost_scores = cross_val_score(adaboost_model, X, y, cv=k)

# XGBoost
xgb_model = XGBClassifier()
xgb_scores = cross_val_score(xgb_model, X, y, cv=k)

# LightGBM
lgbm_model = LGBMClassifier()
lgbm_scores = cross_val_score(lgbm_model, X, y, cv=k)

# Imprimir los resultados de cada modelo
print("Naive Bayes:", nb_scores.mean())
print("LDA:", lda_scores.mean())
print("Regresión logística:", logreg_scores.mean())
print("SVM:", svm_scores.mean())
print("Árboles de decisión:", dt_scores.mean())
print("Random Forest:", rf_scores.mean())
print("Análisis de discriminante lineal:", lda_scores.mean())
print("Análisis de discriminante cuadrático:", qda_scores.mean())
print("AdaBoost:", adaboost_scores.mean())
print("XGBoost:", xgb_scores.mean())
print("LightGBM:", lgbm_scores.mean())
