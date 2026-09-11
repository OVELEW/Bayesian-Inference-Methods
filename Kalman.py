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
        predicted_z = self.F @ self.z
        predicted_Sigma = (self.F @ self.Sigma @ self.F.T + self.Q)
        predicted_y = self.H @ predicted_z
        S = self.H @ predicted_Sigma @ self.H.T + self.R
        K = predicted_Sigma @ self.H.T @ jnp.linalg.inv(S)
        self.z = predicted_z + K @ (y-predicted_y)
        self.Sigma = predicted_Sigma - K @ S @ K.T

dt = 0.5
F = np.array([[1 , dt],
              [0 , 1]])

q = 2
Q = np.array([[q, 0],
              [0, q]])

H = np.array([[1, 0],
              [0, 1]])

r = 5
R = np.array([[r, 0],
              [0, r]])

z_0 = np.array([[0],
                [0]])

c = 100
C_0 = np.array([[c, 0],
                [0, c]])

KF = KalmanFilter(F, Q, H, R, z_0, C_0)
rng = np.random.default_rng()

z_true = np.arange(0,100, dt)
z_noise = rng.normal(loc = 0, scale = np.sqrt(q), size = len(z_true))
z = z_true+z_noise
y = z + rng.normal(loc = 0, scale = np.sqrt(r), size = len(z_true))
z = np.vstack((z, np.full_like(z, 1))).T
y = np.vstack((y, np.full_like(y, rng.normal(loc = 0, scale = np.sqrt(r))))).T
z_estimated = []

for index in y:
    KF.predict_update_steps(index)
    z_estimated.append(KF.z[0]) 

z_estimated = np.array(z_estimated)

plt.plot(z_true, z_estimated[:,0], label = "Kalman")
plt.plot(z_true, z[:,0], label = "True values")
plt.plot(z_true, y[:,0], label = "Measurements")
plt.legend()
plt.grid()
plt.show()
print(z)
