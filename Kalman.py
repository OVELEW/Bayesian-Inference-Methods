import jax
import jax.numpy as jnp

class KalmanFilter:
    def __init__(self, F, Q, H, R, z_0, C_0):
        self.F = F # Transition matrix
        self.Q = Q # Covariance matrix of prediction
        self.H = H # Measurement matrix
        self.R = R # Covariance matrix of measurement
        self.z = z_0 # Initial state estimate
        self.Sigma = C_0 # Covariance of the initial state estimate

    def predict_update_steps(self, y):
        predicted_z = self.F @ self.z
        predicted_Sigma = (self.F @ self.Sigma @ self.F.T + self.Q)
        predicted_y = self.H @ predicted_z
        S = self.H @ predicted_Sigma @ self.H.T + self.R
        K = predicted_Sigma @ self.H.T @ jnp.linalg.inv(S)
        self.z = self.z + K@(y-predicted_y)
        self.Sigma = predicted_Sigma - K @ S @ K.T
