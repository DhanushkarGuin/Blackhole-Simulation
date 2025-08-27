import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Data Preparation --- #
import pandas as pd
dataset = pd.read_csv('ray_dataset.csv')

features = dataset.columns
new_features = features.drop(['ray_id','step','x','y'])
# print(new_features)

X = dataset[new_features].values[:-1] # Current State
y = dataset[new_features].values[1:] # Next state

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.fit_transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.fit_transform(y_test)

# --- Neural ODE modeling --- #

import tensorflow as tf
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
from tensorflow.keras import layers,models

model = models.Sequential([
    layers.Input(shape=(len(new_features),)),
    layers.Dense(64, activation='tanh'),
    layers.Dense(64, activation='tanh'),
    layers.Dense(len(new_features))
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

history = model.fit(
    X_train,y_train,
    validation_data=(X_test,y_test),
    epochs=20,
    batch_size=64
)

loss = model.evaluate(X_test, y_test)
print(f"Test MSE: {loss}")

# --- Exporting the working model --- #
import joblib

model.save('neuralODE_model.keras')
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print('Exportation done!')