import numpy as np

class KalmanFilter:
    def __init__(self, F, Q, H, R, z_0, C_0):
        self.F = np.array(F) # Transition matrix
        self.Q = np.array(Q) # Covariance matrix of process noise
        self.H = np.array(H) # Measurement matrix
        self.R = np.array(R) # Covariance matrix of measurement noise
        self.z = np.array(z_0) # Initial state estimate
        self.z_predicted = None
        self.Sigma_predicted = None
        self.Sigma = np.array(C_0) # Covariance of the initial state estimate

    def predict_update_steps(self, y):
        #Calculation of one-step-ahead values
        self.z_predicted = self.F @ self.z
        self.Sigma_predicted = (self.F @ self.Sigma @ self.F.T + self.Q)
        y_predicted = self.H @ self.z_predicted

        #Calculation of Kalman gain matrix
        S = self.H @ self.Sigma_predicted @ self.H.T + self.R

        #The calculations avoid using unstable inverse of S matrix
        #Here we use the fact that S^T @ K^T = H @ self.Sigma_predicted^T, and that self.Sigma_predicted 
        # and S are covariance matrices which are symmetric
        K = np.linalg.solve(S,self.H @ self.Sigma_predicted).T

        #Corrected mean value and covaraince matrix
        self.z = self.z_predicted + K @ (y-y_predicted)
        #Calculation of updated covariance matrix using Joseph form for numerical stability
        I_KH = np.eye(len(self.z)) - K @ self.H
        self.Sigma = I_KH @ self.Sigma_predicted @ I_KH.T + K @ self.R @ K.T