import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# Carregar os dados do CSV
dados = pd.read_csv("dados.csv", sep=',')

# Selecionar os dados de entrada (X) e a coluna alvo (Y)
X = dados[['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol', 'fasting blood sugar', 'max heart rate', 'exercise angina', 'ST slope']]
Y = dados[['target']]

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, test_size=0.1, random_state=0)

# Padronizar os dados
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Treinar o modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train.values.ravel())  # Ajustar Y_train com .ravel()

# Fazer previsões no conjunto de teste
y_pred = knn.predict(X_test)

# Avaliar o modelo
print(f"Acurácia: {metrics.accuracy_score(Y_test, y_pred)}")

# Salvar o modelo treinado
joblib.dump(knn, 'modelo_knn.pkl')
print("Modelo treinado e salvo como 'modelo_knn.pkl'")

