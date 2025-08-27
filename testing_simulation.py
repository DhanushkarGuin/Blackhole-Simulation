import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Load trained model and scaler
model = load_model("neuralODE_model.keras")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

def simulate_ray(initial_state, steps=100):
    state = scaler_X.transform(initial_state.reshape(1, -1))
    trajectory = []
    for _ in range(steps):
        next_state = model.predict(state, verbose=0)

        denorm_next_state = scaler_y.inverse_transform(next_state)[0]
        trajectory.append(denorm_next_state)
        state = next_state
    return np.array(trajectory)

# Example usage
init_state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
trajectory = simulate_ray(init_state, steps=200)
print("Trajectory shape:", trajectory.shape)