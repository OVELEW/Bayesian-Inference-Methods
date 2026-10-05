import Kalman.KalmanFilter as kf
import Kalman.KalmanSmoother as ks
import numpy as np
import matplotlib.pyplot as plt

dt = 0.5
F = np.array([[1 , dt],
              [0 , 1]])

q = 1
Q = np.array([[q, 0],
              [0, q]])

H = np.array([[1, 0],
              [0, 1]])

r = 3
R = np.array([[r, 0],
              [0, r]])

z_0 = np.array([0, 1])

c = 1
C_0 = np.array([[c, 0],
                [0, c]])

KF = kf.KalmanFilter(F, Q, H, R, z_0, C_0)
KS = ks.KalmanSmoother(F)
rng = np.random.default_rng()

T = 20
time = np.arange(0, T, dt)
stages = len(time)
z_true = np.zeros((stages, 2))

#Producing noise of measurement and process
process_noise = rng.multivariate_normal(mean=[0, 0], cov=Q, size=stages-1)
measurement_noise = rng.multivariate_normal(mean=[0, 0], cov = R, size = stages)

for t in range(1, stages):
    z_true[t] = F @ z_true[t-1] + process_noise[t-1]

y = z_true + measurement_noise

z_updated = []
Sigma_updated = []
z_predicted = []
Sigma_predicted = []
for index in y[1:]: 
    KF.predict_update_steps(index)
    z_updated.append(KF.z)
    Sigma_updated.append(KF.Sigma)
    z_predicted.append(KF.z_predicted)
    Sigma_predicted.append(KF.Sigma_predicted)

Sigma_updated = np.array(Sigma_updated)
z_updated = np.array(z_updated)
z_predicted = np.array(z_predicted)
Sigma_predicted = np.array(Sigma_predicted)

z_smoothed, Sigma_smoothed = KS.smooth_steps(z_predicted, Sigma_predicted, z_updated, Sigma_updated)

plt.plot(time[1:], z_smoothed[:,0], label = "Smoothed values")
plt.plot(time[1:], z_true[1:,0], label = "True values")
plt.plot(time[1:], y[1:,0], label = "Measured values")
plt.plot(time[1:], z_updated[:,0], label="Kalman")
plt.legend()
plt.grid()
plt.show()