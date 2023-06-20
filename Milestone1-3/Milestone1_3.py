from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


class Simulation(object):
    """
    Creates a simulation object

    """

    def __init__(self, f=None, x_range=15, y_range=10, t_range=100, omega=0.1, seed=None):
        super(Simulation, self).__init__()

        # density function f(v, x, y) with size
        # (velocity direction: 9, x position: x_range, y position: y_range)
        if f is not None:
            # if a certain probability distribution is given
            self.f = f
            x_range = f.shape[1]
            y_range = f.shape[2]
        else:
            # else initialize it randomly
            # set a random number seed if given
            if seed:
                np.random.seed(seed)
            self.f = np.random.uniform(0, 1, size=(9, x_range, y_range))

        # norm it to one
        self.f /= self.f.sum()

        # time array
        self.t = np.arange(t_range)

        # velocity directions
        self.c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                           [1, 1], [-1, 1], [-1, -1], [1, -1]])

        # The collision frequency
        self.omega = omega

        # track the whole distribution function f
        self.f_t = np.empty((t_range, 9, x_range, y_range))

        # do the propagation
        self.propagate()

    def rho(self):
        """
        Calculates the particle density rho(x, y) with shape (x_range, y_range)
        for a given probability density f(v, x, y)
        """
        return np.sum(self.f, axis=0)

    def v(self):
        """
        Calculates the velocity density v(x, y) with shape (2, x_range, y_range)
        for a given probability density f(v, x, y)
        """
        current = np.einsum('cxy, ci -> ixy', self.f, self.c)
        rho = self.rho()
        return np.divide(current, rho, out=np.zeros_like(current), where=rho != 0,
                         casting='unsafe')

    def streaming(self):
        """
        Shifts the positions in the probability density f(v, x, y) in the given direction
        direction is multiplied with (1, -1) to make 'up' the positive y-direction

        """
        # velocity is multiplied with (1, -1) to make the positive y-direction pointing upwards
        for i, direction in enumerate(self.c):
            self.f[i] = np.roll(self.f[i], direction * (1, -1), axis=(1, 0))

    def f_eq(self):
        w = np.array([16, 4, 4, 4, 4, 1, 1, 1, 1]) / 36
        v = self.v()
        # w * rho
        w_rho = np.einsum('c, xy -> cxy', w, self.rho())
        # c * v
        c_v = np.einsum('ci, ixy -> cxy', self.c, v)
        # |v|^2
        v2 = np.abs(np.sum(v, axis=0))**2
        return w_rho * (1 + 3 * c_v + 9 / 2 * c_v**2 - 3 / 2 * v2)

    def collision(self):
        """
        Collides the particles with the probability density f(v, x, y) according to the BTE
        """
        self.f += self.omega * (self.f_eq() - self.f)

    def propagate(self):
        """
        Propagates the density f in time and stores the positions in rho_t
        """
        for t in self.t:
            # save the current step of f
            self.f_t[t] = self.f
            # do a streaming step
            self.streaming()
            # do a collision step
            self.collision()

    def calc_rho_t(self):
        """
        Returns the particle density at every timestep
        Shape: (t_range, x_range, y_range)
        """
        return np.sum(self.f_t, axis=1)

    def calc_v_t(self):
        """
        Returns the particle density at every timestep
        Shape: (2, t_range, x_range, y_range)
        """
        current_t = np.einsum('tcxy, ci -> itxy', self.f_t, self.c)
        rho_t = self.calc_rho_t()
        return np.divide(current_t, rho_t, out=np.zeros_like(current_t), where=rho_t != 0,
                         casting='unsafe')

    def save_rho_t(self, filename):
        np.save(filename, self.calc_rho_t)

    def save_v_t(self, filename):
        np.save(filename, self.calc_v_t)

    def save_f_t(self, filename):
        np.save(filename, self.f_t)

    def streamplot(self, save=False):
        x_range = self.f.shape[1]
        y_range = self.f.shape[2]
        fig, ax = plt.subplots()
        ax.set_xlim(-0.5, x_range + 0.5)
        ax.set_ylim(-0.5, y_range + 0.5)

        y, x = np.mgrid[:x_range, :y_range]
        v = self.v()
        ux = v[0]
        uy = v[1]
        plt.streamplot(x, y, ux, uy)
        if save:
            plt.savefig(save)
        plt.show()

    def animate_streamplot(self, interval=100, save=False):
        x_range = self.f.shape[1]
        y_range = self.f.shape[2]
        y, x = np.mgrid[:x_range, :y_range]
        v_t = self.calc_v_t()
        ux_t = v_t[0]
        uy_t = v_t[1]
        fig, ax = plt.subplots()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # ax.set_xlim(-0.5, x_range + 0.5)
        # ax.set_ylim(-0.5, y_range + 0.5)
        vmax = np.max(self.f_t)
        # color according to velocity
        color = np.sqrt(ux_t[0]**2 + uy_t[0]**2)
        stream = ax.streamplot(x, y, ux_t[0], uy_t[0], color=color, cmap='jet', arrowsize=1)

        def animate(i):
            # Clear lines
            for artist in ax.collections:
                artist.remove()
            # Clear arrowheads streamplot.
            for artist in ax.get_children():
                if isinstance(artist, FancyArrowPatch):
                    artist.remove()
            # color according to velocity
            color = np.sqrt(ux_t[i]**2 + uy_t[i]**2)
            stream = ax.streamplot(x, y, ux_t[i], uy_t[i], color=color, cmap='jet', arrowsize=1)
            return stream

        anim = animation.FuncAnimation(fig, animate, frames=ux_t.shape[0],
                                       interval=interval, blit=False)
        if save:
            writer = animation.FFMpegWriter(fps=20)
            anim.save(filename=save, writer=writer)
        plt.show()
        plt.close()

    def animate_rho_t(self, interval=100, save=False):
        fig, ax = plt.subplots()
        ims = []
        rho_t = self.calc_rho_t()
        vmax = np.max(rho_t)
        for t in self.t:
            im = ax.imshow(rho_t[t], animated=True, vmin=0, vmax=vmax)
            if t == 0:
                ax.imshow(rho_t[t])  # show an initial one instead of white background
            ims.append([im])
        fig.colorbar(ims[0][0])
        ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True)
        if save:
            writer = animation.FFMpegWriter(fps=20)
            ani.save(filename=save, writer=writer)
        plt.show()
        plt.close()

    def animate_f_t(self, interval=100, save=False):
        fig, ax = plt.subplots(3, 3)
        vmax = np.max(self.f_t)
        ims = []
        # permutation of c such that it is displayed on the correct position
        c_permutation = [6, 3, 7, 2, 0, 4, 5, 1, 8]

        for t in self.t:
            temp_ims = []
            for i in range(3):
                for j in range(3):
                    pos = c_permutation[i + 3 * j]
                    im = ax[i, j].imshow(self.f_t[t, pos], animated=True, vmin=0, vmax=vmax)
                    if t == 0:
                        # show an initial one instead of white background
                        ax[i, j].imshow(self.f_t[t, pos])
                    temp_ims.append(im)
            ims.append(temp_ims)
        ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True)
        fig.colorbar(ims[0][0], ax=ax)
        if save:
            writer = animation.FFMpegWriter(fps=20)
            ani.save(filename=save, writer=writer)
        plt.show()
        plt.close()
