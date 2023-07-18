from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import Normalize
from matplotlib import colormaps


class Simulation(object):
    """
    Creates a simulation object

    """

    def __init__(self, f=None, x_range=15, y_range=10, t_range=100, omega=0.1,
                 dry_nodes=None, dry_velocities=None, seed=None, p_out=2, delta_p=1):
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

        # velocity directions
        self.c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                           [1, 1], [-1, 1], [-1, -1], [1, -1]])

        # set f in the equilibrium state at some temperature
        self.f = self.f_eq()

        # time array
        self.t = np.arange(t_range)

        # direction weights
        self.w = np.array([16, 4, 4, 4, 4, 1, 1, 1, 1]) / 36

        # anti-channels
        self.anti_c = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

        # The collision frequency
        self.omega = omega

        # Pressure values
        self.p_out = p_out
        self.delta_p = delta_p

        # The dry nodes grid
        # If not given, then don't place any walls
        if dry_nodes is None:
            self.dry_nodes = np.zeros((x_range, y_range), dtype=bool)
        # else check if dimensions are right
        else:
            if dry_nodes.shape[0] != x_range or dry_nodes.shape[1] != y_range:
                raise ValueError("Dimensions of dry_nodes doesn't have the right dimensions")
            self.dry_nodes = dry_nodes

        # The dry velocities grid
        # If not given, then don't initialize any velocities
        if dry_velocities is None:
            self.dry_velocities = np.zeros((2, x_range, y_range))
        # else check if dimensions are right
        else:
            if (dry_velocities.shape[0] != 2 or dry_velocities.shape[1] != x_range or
                    dry_velocities.shape[2] != y_range):
                raise ValueError("Dimensions of dry_velocities doesn't have the right dimensions")
            self.dry_velocities = dry_velocities
            # set velocities to zero where there is a wet node according to dry_nodes
            self.dry_velocities = np.where(self.dry_nodes, self.dry_velocities, 0)

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

    def handle_walls(self):
        """
        Shifts the positions of the probability density f(v, x, y) according to the given
        boundaries with dry_nodes and dry_velocities
        """
        # Copy f at dry nodes
        f_copy = np.where(self.dry_nodes, self.f, np.nan)
        # Add moving to the walls
        rho_w = np.average(self.rho())
        c_v = np.einsum('ci, ixy -> cxy', self.c, self.dry_velocities)
        w_c_v = np.einsum('c, cxy -> cxy', self.w, c_v)
        # 2 / cs**2 = 6, with cs = sqrt(1/3)
        f_copy -= 6 * rho_w * w_c_v

        # Flip velocity channels
        f_copy = f_copy[self.anti_c]
        # Do streaming in copied f
        for i, direction in enumerate(self.c):
            f_copy[i] = np.roll(f_copy[i], direction * (1, -1), axis=(1, 0))
        # Replace the entries which are not nan in the copy in the original f
        np.copyto(self.f, f_copy, where=~np.isnan(f_copy))

    def poisson_flow(self):
        # cs^2 = 1/3
        rho_in = np.ones((self.x_range, self.y_range)) * (self.p_out - self.delta_p) * 3
        rho_out = np.ones((self.x_range, self.y_range)) * self.p_out * 3
        self.f[:, :, 0] = self.f_eq(rho_in)[:, :, -2] + (self.f[:, :, -2] - self.f_eq[:, :, -2])
        self.f[:, :, -1] = self.f_eq(rho_out)[:, :, 1] + (self.f[:, :, 1] - self.f_eq[:, :, 1])

    def f_eq(self, rho=None, v=None):
        w = np.array([16, 4, 4, 4, 4, 1, 1, 1, 1]) / 36
        if rho is None:
            rho = self.rho()
        if v is None:
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
            # apply boundary conditions (walls)
            self.handle_walls()
            # do a collision step
            self.collision()
            # do poisson flow
            self.poisson_flow()

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

    def streamplot(self, t=0, save=False):
        x_range = self.f.shape[1]
        y_range = self.f.shape[2]
        y, x = np.mgrid[:x_range, :y_range]

        v_t = self.calc_v_t()
        ux = v_t[0, t]
        uy = v_t[1, t]

        fig, ax = plt.subplots()
        ax.set_xlim(-0.5, y_range + 0.5)
        ax.set_ylim(-0.5, x_range + 0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        vmax = np.max(v_t)
        norm = Normalize(vmin=0, vmax=vmax, clip=False)

        # color according to velocity
        # there has to be a certain amount of nonzero numbers somehow
        if (np.nonzero((v_t[0, t] + v_t[1, t])[~self.dry_nodes])[0].size
                / (x_range * y_range) > 0.02):
            color = np.sqrt(ux**2 + uy**2)
        else:
            color = colormaps['viridis'].colors[0]
        # Create a mask for dry nodes
        mask = np.array(self.dry_nodes, dtype=bool)
        ux = np.ma.array(ux, mask=mask)

        ax.streamplot(x, y, ux, uy, color=color, norm=norm)
        ax.imshow(~mask, interpolation='nearest', cmap='gray')

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
        ax.set_xlim(-0.5, y_range + 0.5)
        ax.set_ylim(-0.5, x_range + 0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        vmax = np.max(v_t)
        norm = Normalize(vmin=0, vmax=vmax, clip=False)
        # color according to velocity
        # there has to be a certain amount of nonzero numbers somehow
        if (np.nonzero((v_t[0, 0] + v_t[1, 0])[~self.dry_nodes])[0].size
                / (x_range * y_range) > 0.02):
            color = np.sqrt(ux_t[0]**2 + uy_t[0]**2)
        else:
            color = colormaps['viridis'].colors[0]

        # Create a mask for dry nodes
        mask = np.array(self.dry_nodes, dtype=bool)
        ux_0 = np.ma.array(ux_t[0], mask=mask)
        uy_0 = np.ma.array(uy_t[0], mask=mask)
        ax.streamplot(x, y, ux_0, uy_0, color=color, norm=norm)

        # plot dry_nodes them as black boxes
        ax.imshow(~mask, interpolation='nearest', cmap='gray')

        def animate(i):
            # Clear lines
            for artist in ax.collections:
                artist.remove()
            # Clear arrowheads streamplot.
            for artist in ax.get_children():
                if isinstance(artist, FancyArrowPatch):
                    artist.remove()
            # color according to velocity
            # there has to be a certain amount of nonzero numbers somehow
            if (np.nonzero((v_t[0, i] + v_t[1, i])[~self.dry_nodes])[0].size
                    / (x_range * y_range) > 0.02):
                color = np.sqrt(ux_t[i]**2 + uy_t[i]**2)
            else:
                color = colormaps['viridis'].colors[0]
            ux_i = np.ma.array(ux_t[i], mask=mask)
            uy_i = np.ma.array(uy_t[i], mask=mask)
            stream = ax.streamplot(x, y, ux_i, uy_i, color=color, norm=norm)
            # # plot dry_nodes them as black boxes
            ax.imshow(~mask, interpolation='nearest', cmap='gray')
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
