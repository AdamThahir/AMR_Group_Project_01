import numpy as np
from numpy import dot, tile, linalg, log, exp
from numpy.linalg import inv, det

class KalmanFilter:
    # Main concepts obtained from https://arxiv.org/pdf/1204.0375.pdf
    def __init__(self):
        dt = 0.01
        sensor_count = 3
        self.X = np.matrix([[0.0], [0.0], [0.0]])
        self.P = np.matrix(np.identity(self.X.shape[0]))
        self.A = np.matrix(np.identity(self.X.shape[0]))

        self.U = np.matrix(np.zeros((self.X.shape[0],1)))
        self.H = np.matrix(np.identity(self.X.shape[0]))
        self.R = np.matrix(np.identity(sensor_count))
        self.I = np.matrix(np.identity(self.X.shape[0]))

        self.R *= dt
        self.is_first = True


    def predict (self):
        X = self.A * self.X + self.U
        P = self.A * self.P * self.A.T

        self.X = X
        self.P = P

        return X

    def update(self, Z):
        if (self.is_first):
            self.X = Z
            self.is_first = False

        # Z contains the sensor values

        W = Z - self.H * self.X
        S = self.H * self.P * self.H.T + self.R
        K = self.P * self.H.T * S.I
        X = self.X + K*W
 
        self.X = X

        P = (self.I - K * self.H) * self.P
        self.P = P
