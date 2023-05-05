from matplotlib import animation
from IPython.display import HTML
import numpy as np
import matplotlib.pyplot as plt


class Simulation(object):
    """
    Creates a simulation object

    """

    def __init__(self, x_range=15, y_range=10, t_range=100, seed=None):
        super(Simulation, self).__init__()

        # density function f(v, x, y) with size
        # (velocity direction: 9, x position: x_range, y position: y_range)
        self.f = np.random.uniform(0, 1, size=(9, x_range, y_range))
        # norm it to one
        self.f /= self.f.sum()

        # time array
        self.t = np.arange(t_range)

        # velocity directions
        self.c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                           [1, 1], [-1, 1], [-1, -1], [1, -1]])

        # track the positions (density, rho(t, x, y)) for animations later
        self.rho_t = np.empty((t_range, x_range, y_range))

        # set a random number seed if given
        if seed:
            np.random.seed(seed)

    def rho(self):
        """
        Calculates the particle density rho(x, y) for a given probability density f(v, x, y)
        """
        return np.sum(self.f, axis=0)

    def v(self):
        """
        Calculates the velocity density v(x, y) for a given probability density f(v, x, y)
        """
        return ((1 / self.rho())[..., np.newaxis] *
                np.sum((self.f[..., np.newaxis] * self.c[:, np.newaxis, np.newaxis, :]), axis=0))

    def streaming(self):
        """
        Shifts the positions in the probability density f(v, x, y) in the given direction
        direction is multiplied with (1, -1) to make 'up' the positive y-direction

        """
        # velocity is multiplied with (1, -1) to make the positive y-direction pointing upwards
        for i, direction in enumerate(self.c):
            self.f[i] = np.roll(self.f[i], direction * (1, -1), axis=(1, 0))

    def propagate(self):
        """
        Propagates the density f in time and stores the positions in rho_t
        """
        for t in self.t:
            # choose in which direction to go
            self.rho_t[t] = self.rho()
            self.streaming()

    def animate_rho_t(self, save=False):
        fig, ax = plt.subplots()
        ims = []
        for i in self.t:
            im = ax.imshow(self.rho_t[i], animated=True)
            if i == 0:
                ax.imshow(self.rho_t[i])  # show an initial one instead of white background
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
        if save:
            writer = animation.FFMpegWriter(fps=20)
            ani.save("flow.gif", writer=writer)
        plt.show()
        plt.close()


sim = Simulation(t_range=100, x_range=11, y_range=13)
sim.propagate()
sim.animate_rho_t(save=False)
