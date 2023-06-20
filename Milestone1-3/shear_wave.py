import pdb
from Milestone1_3 import Simulation
import numpy as np
import matplotlib.pyplot as plt


def laplace(m):
    result = np.zeros_like(m)
    for axis in range(len(m.shape)):
        forward = np.roll(m, 1, axis=axis)
        backward = np.roll(m, -1, axis=axis)
        result += forward - 2 * m + backward
    return result


x_range = 200
y_range = 10

# density for shear wave decay
rho_0 = 0.5
epsilon = 0.4

# roadmap point 1

rho_x = rho_0 + epsilon * np.sin(2 * np.pi * np.arange(x_range) / x_range)
rho = np.einsum('x, y -> xy', rho_x, np.ones(y_range))
f_1 = np.zeros(shape=(9, x_range, y_range))
f_1[0] = rho
sim_1 = Simulation(f=f_1, omega=0.02, t_range=200)
# sim_1.animate_f_t(interval=50)
# sim_1.animate_rho_t(interval=50)  # , save="output/shear_wave_rho.gif")
rho_t = sim_1.calc_rho_t()[:, :, 0]
fft = np.fft.fft(rho_t)
freq = np.fft.fftfreq(rho_t.shape[-1])
init_freq_idx = np.argmax(fft[0].imag)

plt.plot(fft[:, init_freq_idx].imag)
plt.xlabel("Time")
plt.ylabel("Highest amplitude")
plt.savefig("output/highest_amplitude_1.pdf")
plt.show()

# for i in range(0, 50, 5):
#     plt.plot(rho_t[i], label=f"time={i}")
# plt.legend()
# plt.show()

# roadmap point 2

x_range = 10
y_range = 200

# rho = np.ones(shape=(x_range, y_range))
# v_x = epsilon * np.sin(2 * np.pi * np.arange(y_range) / y_range)
# v_max = np.max(np.abs(v_x))
# v_pos = np.abs(np.where(v_x > 0, v_x, 0))
# v_neg = np.abs(np.where(v_x < 0, v_x, 0))
# v_neu = (np.ones_like(v_x) * v_max) - (v_pos + v_neg)
# f_2 = np.zeros(shape=(9, x_range, y_range))
# f_2[1] = v_neg
# f_2[3] = v_pos
# f_2[0] = v_neu
# sim_2 = Simulation(f=f_2, omega=0.04, t_range=200)
# v_x_t = sim_2.calc_v_t()[0, :, 0, :]
# fft = np.fft.fft(v_x_t)
# freq = np.fft.fftfreq(v_x_t.shape[-1])
# init_freq_idx = np.argmax(fft[0].imag)
# for i in range(0, 50, 5):
#     plt.plot(v_x_t[i], label=f"time={i}")
# plt.legend()
# plt.xlabel("y")
# plt.ylabel("$u_x(y)$")
# plt.savefig("output/v_x_t.pdf")
# plt.show()

# sim_2.animate_f_t(interval=50, save='output/shear_wave_f_2.gif')
# sim_2.animate_rho_t(interval=50, save="output/shear_wave_rho2.gif")
