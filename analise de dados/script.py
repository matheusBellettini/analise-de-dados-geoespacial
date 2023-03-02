# Importação das bibliotecas necessárias
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Carregamento dos dados geoespaciais
data = pd.read_csv('dados_geoespaciais.csv')

# Pré-processamento dos dados
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Criação do modelo de classificação de imagens de satélite
model = keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compilação e treinamento do modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Avaliação do modelo
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Previsão de áreas de risco
risk_areas = model.predict(X)
risk_areas = (risk_areas > 0.5)
risk_areas = np.array(risk_areas, dtype=int)
data['risk_area'] = risk_areas
data.to_csv('dados_geoespaciais_pred.csv', index=False)

# Detecção de anomalias
anomaly_detection_model = keras.Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(128, activation='relu'),
    Dense(1, activation='linear')
])
anomaly_detection_model.compile(optimizer='adam', loss='mse')
anomaly_detection_model.fit(X_train, y_train, epochs=10, batch_size=32)
anomaly_scores = anomaly_detection_model.predict(X)
anomaly_scores = np.squeeze(anomaly_scores)
data['anomaly_score'] = anomaly_scores
data.to_csv('dados_geoespaciais_pred.csv', index=False)
