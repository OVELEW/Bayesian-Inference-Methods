from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import src.Kalman as kl
import numpy as np
import matplotlib.pyplot as plt


dt = 0.5
F = np.array([[1 , dt],
              [0 , 1]])

q = 1
Q = np.array([[q, 0],
              [0, 0]])

H = np.array([[1, 0],
              [0, 1]])

r = 1
R = np.array([[r, 0],
              [0, r]])

z_0 = np.array([[0],
                [0]])

c = 100
C_0 = np.array([[c, 0],
                [0, c]])

KF = kl.KalmanFilter(F, Q, H, R, z_0, C_0)
rng = np.random.default_rng()

T = 100
time = np.arange(0, T, dt)
stages = len(time)
z_true = np.zeros((stages, 2))

z_true[0] = [0,1]

#Producing noise of measurement and process
process_noise = rng.multivariate_normal(mean=[0, 0], cov=Q, size=stages-1)
measurement_noise = rng.multivariate_normal(mean=[0,0], cov = R, size = stages)

for t in range(1, stages):
    z_true[t] = F @ z_true[t-1] + process_noise[t-1]

y = z_true + measurement_noise

z_estimated = [[1,0]]
for index in y[1:]: 
    KF.predict_update_steps(index) 
    z_estimated.append(KF.z[0]) 
z_estimated = np.array(z_estimated)

plt.plot(time, z_true[:,0], label = "True values")
plt.plot(time, y[:,0], label = "Measured values")
plt.plot(time, z_estimated[:,0], label="Kalman")
plt.legend()
plt.show()
