# Importando as bibliotecas necessárias:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import numpy as np
import joblib

df = pd.read_csv('dados.csv')

labelencoder = LabelEncoder()
df = df.apply(labelencoder.fit_transform)

# X = dados de entrada Y = classe
Y = df['target'].values
X = df.drop(columns = ['target'])

X_train, X_test, y_train, y_test = train_test_split(X,Y, shuffle = True, test_size=0.1,random_state=0)
# Verificando as formas dos dados:
X_train.shape,X_test.shape

y_train.shape,y_test.shape

# Instânciando o objeto classificador:
clf = DecisionTreeClassifier()

# Treinando o modelo de arvore de decisão:
clf = clf.fit(X_train,y_train)

# Verificando as features mais importantes para o modelo treinado:
clf.feature_importances_

# Salvar o modelo treinado
joblib.dump(clf, 'decision_tree.pkl')
print("Modelo treinado e salvo como 'decision_tree.pkl'")

