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

        # self.A = np.array([[1, 0, dt , 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        # self.Q = np.eye(self.X.shape[0])
        # self.B = np.eye(self.X.shape[0])
        self.U = np.matrix(np.zeros((self.X.shape[0],1)))
        self.H = np.matrix(np.identity(self.X.shape[0]))
        self.R = np.matrix(np.identity(sensor_count))
        self.I = np.matrix(np.identity(self.X.shape[0]))

        self.R *= dt
        self.is_first = True

        #self.n_iter = 50

    def predict (self):
        # X = dot(self.A, self.X) + dot(self.B, self.U)
        X = self.A * self.X + self.U
        # P = dot(self.A, dot(self.P, self.A.T)) + self.Q
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

    def gauss_df(self, X, M, S):
        if M.shape()[1] == 1:
            DX = X - tile(M, X.shape()[1])
            E = 0.5 * np.sum(DX * (dot(inv(S), DX)), axis=0)
            E = E + 0.5 * M.shape()[0] * log(2 * np.pi) + 0.5 * log(det(S))
            P = exp(-E)
        elif X.shape()[1] == 1:
            DX = tile(X, M.shape()[1]) - M
            E = 0.5 * np.sum(DX * (dot(inv(S), DX)), axis=0)
            E = E + 0.5 * M.shape()[0] * log(2 * np.pi) + 0.5 * log(det(S))
            P = exp(-E)
        else:
            DX = X - M
            E = 0.5 * dot(DX.T, dot(inv(S), DX))
            E = E + 0.5 * M.shape()[0] * log(2 * np.pi) + 0.5 * log(det(S))
            P = exp(-E)
        
        return (P[0], E[0])