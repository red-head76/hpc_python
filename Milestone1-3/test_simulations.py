from Milestone1_3 import Simulation
import numpy as np

x_range = 15
y_range = 10

# First test flow in 3 directions
# omega = 0 effectively turns off collisions
rho_1 = np.zeros((x_range, y_range))
rho_1[x_range // 2, y_range // 2] = 1
f_1 = np.zeros(shape=(9, x_range, y_range))
# place "4 particles" in the density: one standing, one flying upwards, one flying to the right and
# one flying diagonally right upwards
f_1[0:3] = rho_1
f_1[5] = rho_1
sim_1 = Simulation(f=f_1, t_range=10, omega=0.0, x_range=15, y_range=10)

# sim_1.animate_streamplot(save="output/test1_stream.gif")
# sim_1.streamplot(save="output/test1_streamplot.svg")
# sim_1.animate_rho_t(save="output/test1_rho.gif")
# sim_1.animate_f_t(save="output/test1_f.gif")
