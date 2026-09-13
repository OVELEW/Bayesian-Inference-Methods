import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, F, Q, H, R, z_0, C_0):
        self.F = F # Transition matrix
        self.Q = Q # Covariance matrix of process noise
        self.H = H # Measurement matrix
        self.R = R # Covariance matrix of measurement noise
        self.z = z_0 # Initial state estimate
        self.Sigma = C_0 # Covariance of the initial state estimate

    def predict_update_steps(self, y):
        #Calculation of one-step-ahead values
        predicted_z = self.F @ self.z
        predicted_Sigma = (self.F @ self.Sigma @ self.F.T + self.Q)
        predicted_y = self.H @ predicted_z

        #Calculation of Kalman gain matrix
        S = self.H @ predicted_Sigma @ self.H.T + self.R
        K = np.linalg.solve(S,self.H @ predicted_Sigma).T
        
        self.z = predicted_z + K @ (y-predicted_y)
        self.Sigma = predicted_Sigma - K @ S @ K.T