import numpy as np
from numpy import dot, tile, linalg, log, exp
from numpy.linalg import inv, det

import numpy.linalg as la

class KalmanFilter:
    # Main concepts obtained from https://arxiv.org/pdf/1204.0375.pdf
    def __init__(self):
        self.dt = 0.01
        sensor_count = 3
        self.X = np.matrix([[0.0], [0.0], [0.0]])
        self.P = np.matrix(np.identity(self.X.shape[0]))
        self.A = np.matrix(np.identity(self.X.shape[0]))

        self.U = np.matrix(np.zeros((self.X.shape[0],1)))
        self.H = np.matrix(np.identity(self.X.shape[0]))
        self.R = np.matrix(np.identity(sensor_count))
        self.I = np.matrix(np.identity(self.X.shape[0]))

        self.R *= self.dt
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


class EKF_SLAM:
    # Main concepts obtained from https://github.com/DevonMorris
    def __init__(self, initialPosition, nObjects, noise_covariance):

        inf = 1e15

        self.mu = np.zeros(3*nObjects + 3)
        self.mu[0:3] = initialPosition
        self.cov = np.matrix(inf * np.eye(3*nObjects + 3))
        self.cov[0,0] = 0
        self.cov[1,1] = 0
        self.cov[2,2] = 0

        self.seen = [False for i in range(nObjects)]

        # Here objects are landmarks in the map
        self.nObjects = nObjects
        self.R = noise_covariance

        # I want to get the predictions here, but I'm not too sure how this could work.
        # Working on preprocessing some of the information, but not able to figure it out yet.
        objectLocations = np.zeros((self.nObjects, 3))
        objectLocations[:,0] = np.random.uniform(low=-10., high=10., size=self.nObjects)
        objectLocations[:,1] = np.random.uniform(low=-10., high=10., size=self.nObjects)
        objectLocations[:,2] = np.arange(self.nObjects)

        with open('landmarks.csv', 'w') as f:
            content = ""
            for landmark in objectLocations:
                for pos in landmark:
                    content += str(pos) + ','
                content += '\n'
            f.write(content)

        self.objectLocations = objectLocations
        self.alpha = np.pi/4

        self.dt = .1

    def get_U(self):
        dt = self.dt
        t = np.arange(0,50.1, dt)
        v = 1 + .5*np.cos(.4*np.pi*t)
        w = -.2 + 2*np.cos(1.2*np.pi*t)

        U = np.column_stack([v, w])

        return U

    def get_Z(self, X):
        Y = self.objectLocations
        
        Z = []
        
        for i in range(Y.shape[0]):
            z = np.zeros(3)
            z[0] = np.linalg.norm(X[:2] - Y[i,:2])
            z[1] = np.arctan2(Y[i,1] - X[1], Y[i,0] - X[0]) - X[2]
            
            z += np.random.multivariate_normal(np.zeros(3), self.R)
            # wrap relative bearing
            if z[1] > np.pi:
                z[1] = z[1] - 2*np.pi
            if z[1] < -np.pi:
                z[1] = z[1] + 2*np.pi
            z[2] = Y[i,2]
            if np.abs(z[1]) < self.alpha/2:
                Z.append(z)
        
        Z = np.array(Z) 

        return Z

    def filter(self, z, u):
        """filters the slam problem over a single time step"""

        self.mu[0:3] = self.non_linear_state_transition_handler(self.mu[0:3],u)
        Fx = np.matrix(np.zeros((3, 3 * self.nObjects + 3)))
        Fx[0,0] = 1
        Fx[1,1] = 1
        Fx[2,2] = 1
        
        F = np.matrix(np.eye(3 * self.nObjects +3)) + Fx.T * self.jacobian_state_state_handler(self.mu[0:3],u) * Fx
        G = 5 * self.jacobian_state_input_handler(self.mu[0:3], u)
        
        self.cov = F * self.cov * F.T + Fx.T * G * self.input_covariance_handler(u) * G.T * Fx
        if z.size == 0:
            return np.copy(self.mu), np.copy(self.cov)
        
        l = z[:,2]

        for j,s in enumerate(l):
            #Measurement Update
            s = int(s)
            if self.seen[s] == False:
                self.mu[3 * (s + 1)] = self.mu[0] + z[j, 0] * np.cos(z[j, 1] + self.mu[2])
                self.mu[3 * (s + 1) + 1] = self.mu[1] + z[j, 0] * np.sin(z[j, 1] + self.mu[2])
                self.mu[3 * (s + 1) + 2] = s
                self.seen[s] = True
                
            y = self.mu[3 * (s + 1):3 * (s + 1) + 3]
            z_hat = self.non_linear_measurement_model_handler(self.mu[:3], y, u)

            Fxj = np.matrix(np.zeros((6, 3 * self.nObjects + 3)))
            Fxj[0,0] = 1
            Fxj[1,1] = 1
            Fxj[2,2] = 1
            Fxj[3,3 * (s + 1)] = 1
            Fxj[4, 3 * (s + 1) + 1] = 1
            Fxj[5, 3 * (s + 1) + 2] = 1
            
            H = self.jacobian_measurement_state_handler(self.mu[:3],y,u) * Fxj
            K = self.cov * H.T * (la.inv(H * self.cov * H.T + self.R))

            innovation = z[j] - z_hat
            if innovation[1] > np.pi:
                innovation[1] -= 2 * np.pi
            elif innovation[1] < -np.pi:
                innovation += 2 * np.pi
            innovation = np.matrix(innovation).T
            update = np.array(K * innovation).flatten()
            self.mu += update
            self.cov = (np.eye(3 * self.nObjects + 3) - K * H).dot(self.cov)
            
        return np.copy(self.mu), np.copy(self.cov)


    def non_linear_state_transition_handler(self, x, u):
        """function handle nonlinear state transition"""
        xp = np.zeros_like(x)
        v = u[0]
        w = u[1]
        stheta = np.sin(x[2])
        ctheta = np.cos(x[2])
        sthetap = np.sin(x[2] + self.dt*w)
        cthetap = np.cos(x[2] + self.dt*w)

        xp[0] = x[0] - v/w*stheta + v/w*sthetap
        xp[1] = x[1] + v/w*ctheta - v/w*cthetap
        xp[2] = x[2] + w*self.dt
        return xp

    def non_linear_measurement_model_handler(self, x, y, u):
        """function handle nonlinear measurement model"""
        zp = np.zeros(3)
    
        zp[0] = np.linalg.norm(x[0:2] - y[0:2])
        zp[1] = np.arctan2(y[1] - x[1], y[0] - x[0]) - x[2]
        if zp[1] > np.pi:
            zp[1] -= 2*np.pi
        elif zp[1] < -np.pi:
            zp[1] += 2*np.pi
        zp[2] = y[2]
        return zp

    def jacobian_state_state_handler(self, x, u):
        """function handle Jacobian of state w/ respect to state"""
        n = x.shape[0]
        v = u[0]
        w = u[1]
        stheta = np.sin(x[2])
        ctheta = np.cos(x[2])
        sthetap = np.sin(x[2] + self.dt*w)
        cthetap = np.cos(x[2] + self.dt*w)
        
        F = np.matrix(np.zeros((n,n)))
        F[0,2] = -v/w*ctheta + v/w*cthetap
        F[1,2] = -v/w*stheta + v/w*sthetap
        return F

    def jacobian_state_input_handler(self, x, u):
        """function handle Jacobian of state w/ respect to input"""
        n = x.shape[0]
        k = u.shape[0]
        v = u[0]
        w = u[1]
        stheta = np.sin(x[2])
        ctheta = np.cos(x[2])
        sthetap = np.sin(x[2] + self.dt*w)
        cthetap = np.cos(x[2] + self.dt*w)
        
        G = np.matrix(np.zeros((n,k)))
        G[0,0] = (-stheta + sthetap)/w
        G[0,1] = v*(stheta-sthetap)/(w**2) + v*(ctheta*self.dt)/w
        G[1,0] = (ctheta - cthetap)/w
        G[1,1] = -v*(ctheta - cthetap)/(w**2) + v*(stheta*self.dt)/w
        G[2,1] = self.dt
        return G

    def jacobian_measurement_state_handler(self, x, y, u):
        """function handle Jacobian of measurement w/ respect to state"""
        H = np.matrix(np.zeros((3,6)))
        dx = y[0] - x[0]
        dy = y[1] - x[1]
        q = (y[0] - x[0])**2 + (y[1] - x[1])**2
        sq = np.sqrt(q)
        H[0,0] = -(dx)/sq
        H[0,1] = -(dy)/sq
        H[0,2] = 0
        H[0,3] = dx/sq
        H[0,4] = dy/sq
        H[0,5] = 0
        H[1,0] = dy/q
        H[1,1] = -dx/q
        H[1,2] = -1
        H[1,3] = -dy/q
        H[1,4] = dx/q
        H[1,5] = 0
        H[2,5] = 1
        return H

    def input_covariance_handler(self, u):
        """function handle Covariance of input"""
        k = u.shape[0]
        v = u[0]
        w = u[1]
        Q = np.matrix(np.zeros((k,k)))

        alpha = np.array([.1, .01, .01, .1])
        Q[0,0] = alpha[0]*v**2 + alpha[1]*w**2
        Q[1,1] = alpha[2]*v**2 + alpha[3]*w**2
        return Q



