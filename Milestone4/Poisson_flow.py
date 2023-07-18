import pdb
from Milestone4 import Simulation
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x_range = 100
y_range = 100
t_range = 401

# uniform density
f = np.zeros(shape=(9, x_range, y_range))
f[0] = 1
dry_nodes = np.zeros((x_range, y_range), dtype=bool)
dry_nodes[0] = True
dry_nodes[-1] = True
# dry_nodes[20:60, 50] = True
# dry_nodes[:, -1] = True
dry_velocities = np.zeros((2, x_range, y_range))
dry_velocities[0, -1, :] = 1
dry_velocities[0, 0, :] = 1
sim_1 = Simulation(f=f, omega=0.02, t_range=t_range, dry_nodes=dry_nodes,
                   dry_velocities=dry_velocities)

f_eq = sim_1.f_eq(rho=sim_1.rho(), v=sim_1.v()[:, 2].reshape(2, 1, y_range))
# sim_1.animate_streamplot(interval=50)
# sim_1.animate_f_t()

# sim_1.streamplot(t=10)
# plt.show()

# x = np.arange(x_range)
# y = np.arange(y_range)
# v_t = sim_1.calc_v_t()
# ux_t = v_t[0]
# uy_t = v_t[1]
# fig, ax = plt.subplots()
# for i in np.linspace(0, t_range - 1, 10, dtype=int):
#     plt.plot(x[1:-1], ux_t[i, 1:-1, 0], label=f"Tmestep {i}")
# plt.legend()
# plt.savefig("output/Coutte_flow_velocity_profile.svg")
# plt.show()
