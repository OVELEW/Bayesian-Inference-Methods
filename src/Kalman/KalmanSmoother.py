import numpy as np
import jax.numpy as jnp

class KalmanSmoother:
    def __init__(self, F):
        self.F = jnp.array(F) # Transition matrix

    def smooth_steps(self, z_estimated_seq, Sigma_estimated_seq, z_updated_seq, Sigma_updated_seq):
        z_smoothed_seq = np.zeros_like(z_updated_seq)
        Sigma_smoothed_seq = np.zeros_like(Sigma_updated_seq)

        #The final point
        z_smoothed_seq[-1] = z_updated_seq[-1]
        Sigma_smoothed_seq[-1] = Sigma_updated_seq[-1]

        steps = len(z_updated_seq)
        for step in range(steps-2, -1, -1):
            #Backwards Kalman gain matrix
            J = np.linalg.solve(Sigma_estimated_seq[step+1],self.F@Sigma_updated_seq[step]).T
            z_smoothed_seq[step] = z_updated_seq[step] + J @ (z_smoothed_seq[step+1] - z_estimated_seq[step+1])
            Sigma_smoothed_seq[step] = Sigma_updated_seq[step] + J @ (Sigma_smoothed_seq[step+1] - Sigma_estimated_seq[step+1]) @ J.T
        return z_smoothed_seq, Sigma_smoothed_seq