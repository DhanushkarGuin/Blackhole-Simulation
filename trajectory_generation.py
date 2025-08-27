import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import numpy as np
from tensorflow.keras.models import load_model
import joblib 

class TrajectoryNN:
    def __init__(self,
                model_path='neuralODE_model.keras',
                scaler_X_path='scaler_X.pkl',
                scaler_y_path='scaler_y.pkl'):
        
        self.model = load_model(model_path, compile=False)
        self.scaler_X = joblib.load(scaler_X_path)
        self.scaler_y = joblib.load(scaler_y_path)

    def predict(self, state_vec):
        x = np.asarray(state_vec, dtype=np.float32).reshape(1,-1)
        x_n = self.scaler_X.transform(x)
        y_n = self.model.predict(x_n,verbose=0)
        y = self.scaler_y.inverse_transform(y_n)
        return y[0]
        
    def rollout(self, state_vec,steps=100):
        trajectory = [np.asarray(state_vec, dtype=np.float64)]
        s = trajectory[0].copy()
        for _ in range(steps):
            s = self.predict_next(s)
            trajectory.append(s)
        return np.array(trajectory)