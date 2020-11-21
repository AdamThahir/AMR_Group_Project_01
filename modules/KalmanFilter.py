import numpy as np
from numpy import dot, tile, linalg, log, exp
from numpy.linalg import inv, det

class KalmanFilter:
    # Main concepts obtained from https://arxiv.org/pdf/1204.0375.pdf
    def __init__(self):
        pass

    def predict (self, X, P, A, Q, B, U):
        X = dot(A, X) + dot(B, U)
        P = dot(A, dot(P, A.T)) + Q
        return (X, P)

    def update(self, X, P, Y, H, R):
        IM = dot(H, X)
        IS = R + dot(H, dot(P, H.T))
        K = dot(P, dot(H.T, inv(IS)))
        X = X + dot(K, (Y-IM))
        P = P - dot(K, dot(IS, K.T))
        LH = self.gauss_df(Y, IM, IS)

        return (X, P, K, IM, IS, LH)

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