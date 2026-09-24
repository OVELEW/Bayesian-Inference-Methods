import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

class KalmanSmoother:
    def __init__(self, F, Q):
        self.F = F # Transition matrix
        self.Q = Q # Covariance matrix of process noise
        # self.z_estimated_seq = []
        # self.Sigma_estimated_seq = []
        # self.z_updated_seq = []
        # self.Sigma_updated_seq = []
        # self.z_smoothed_seq = []
        # self.Sigma_smoothed_seq = []

    def smooth_steps(self, z_estimated_seq, Sigma_estimated_seq, z_updated_seq, Sigma_updated_seq):
        self.z_estimated_seq = z_estimated_seq
        self.Sigma_estimated_seq = Sigma_estimated_seq
        self.z_updated_seq = z_updated_seq
        self.Sigma_updated_seq = Sigma_updated_seq
        self.z_smoothed_seq = np.zeros_like(self.z_updated_seq)
        self.Sigma_smoothed_seq = np.zeros_like(self.Sigma_updated_seq)

        #The final point
        self.z_smoothed_seq[-1] = z_updated_seq[-1]
        self.Sigma_updated_seq[-1] = Sigma_updated_seq[-1]

        steps = len(z_updated_seq)
        for step in range(steps-2, -1, -1):
            #Backwards Kalman gain matrix
            J = np.linalg.solve(self.Sigma_estimated_seq[step+1],self.F@self.Sigma_updated_seq[step]).T
            self.z_smoothed_seq[step] = z_updated_seq[step] + J @ (self.z_smoothed_seq[step+1] - self.z_estimated_seq[step+1])
            self.Sigma_smoothed_seq[step] = Sigma_updated_seq[step] + J @ (self.Sigma_smoothed_seq[step+1] - self.Sigma_estimated_seq[step+1]) @ J.T
        return self.z_smoothed_seq, self.Sigma_smoothed_seq