import numpy as np
from numpy.random import uniform, randn
import scipy
from numpy.linalg import norm


class ParticleFilter:
    # Main concepts obtained from https://github.com/Kalman-and-Bayesian-Filters-in-Python/
    def __init__(self):
        self.N = 10
        self.mean = [0,0,0]
        self.median = 0
        self.std = [1,1,1]
        self.iter = 0
        # self.is_first = True
        self.deleted_particles = 0
        self.particles = []
        self.weights = None

        self.particle_mean = None
        self.particle_var = None
        self.R = .1
        self.landmarks = np.array([[0,0], [3,3], [-3,-3], [-3,3], [3,-3], [5,5], [-5,-5], [-5,5], [5,-5]])

        self.create_gaussian_particles()


    def create_uniform_particles(self, x_range, y_range, hdg_range, N):
        particles = np.empty((N, 3))
        particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
        particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
        particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
        particles[:, 2] %= 2 * np.pi

        self.particles = particles
        return particles

    def create_gaussian_particles(self):
        # the particles with given mu and std are created
        # and their weights are initialized
        particles = np.empty((self.N, 3))
        particles[:, 0] = self.mean[0] + (randn(self.N) * self.std[0])
        particles[:, 1] = self.mean[1] + (randn(self.N) * self.std[1])
        particles[:, 2] = self.mean[2] + (randn(self.N) * self.std[2])
        particles[:, 2] %= 2 * np.pi

        self.particles = particles
        self.weights = np.zeros(self.particles.shape[0])
        return particles

    def predict(self, u, dt=1.):       
        # particles' heading are updated
        self.particles[:, 2] += u[0] + (randn(self.N) * self.std[0])
        self.particles[:, 2] %= 2 * np.pi

        # particles move in the direction
        # the heading is given according to robot movement
        dist = (u[1] * dt) + (randn(self.N) * self.std[1])
        self.particles[:, 0] += np.cos(self.particles[:, 2]) * dist
        self.particles[:, 1] += np.sin(self.particles[:, 2]) * dist

        return self.particles

    def update(self, current_pos):
        # the weigths are updated according to the importance (the distance)
        # the closer the particle is, the greater the weight is

        # .1 here is standard error
        # we calculate the distance between robot position to each landmark
        # to be able to assess how good the particles are
        zs = (norm(self.landmarks-[current_pos[0], current_pos[1]],axis=1) + \
                randn(len(self.landmarks))*.1)

        # the weights are calculated according to the position of particles
        # to each landmark
        for i, landmark in enumerate(self.landmarks):
            distance = np.linalg.norm(self.particles[:, 0:2] - landmark, axis=1)
            self.weights *= scipy.stats.norm(distance, self.R).pdf(zs[i])

        self.weights += 1.e-300   
        self.weights /= sum(self.weights)

    def estimate(self):
        # computes the mean and varaince of the particles
        # we plot mean value to compare localization value with the real position
        pos = self.particles[:, 0:2]
        self.particle_mean = np.average(pos, weights=self.weights, axis=0)
        self.particle_var  = np.average((pos - self.particle_mean)**2, weights=self.weights, axis=0)

        return self.particle_mean, self.particle_var

    def neff(self):
        # This is to measure efficiency
        # if this value is lower than our threshold (here we use N/2), then
        # we resample the particles
        return 1. / np.sum(np.square(self.weights))

    def resample_from_index(self, indexes):
        self.particles[:] = self.particles[indexes]
        self.weights.resize(len(self.particles))
        self.weights.fill (1.0 / len(self.weights))

        return self.particles, self.weights
    
    def systematic_resample(self):
        pos = (np.arange(self.N) + randn(self.N)) / self.N
        ins = np.zeros(self.N, 'i') #int
        cumulative_sum = np.cumsum(self.weights)
        i = 0
        j = 0
        while i < self.N and j < self.N:
            if (pos[i] < cumulative_sum[j]):
                ins[i] = j
                i+=1
            else :
                j+=1

        return ins