import pdb
from Milestone1_3 import Simulation
import numpy as np
import matplotlib.pyplot as plt
from copy import copy


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

# rho_x = rho_0 + epsilon * np.sin(2 * np.pi * np.arange(x_range) / x_range)
# rho = np.einsum('x, y -> xy', rho_x, np.ones(y_range))
# f_1 = np.zeros(shape=(9, x_range, y_range))
# f_1[0] = rho
# sim_1 = Simulation(f=f_1, omega=0.02, t_range=200)
# # sim_1.animate_f_t(interval=50)
# # sim_1.animate_rho_t(interval=50)  # , save="output/shear_wave_rho.gif")
# rho_t = sim_1.calc_rho_t()[:, :, 0]
# fft = np.fft.fft(rho_t)
# freq = np.fft.fftfreq(rho_t.shape[-1])
# init_freq_idx = np.argmax(fft[0].imag)

# plt.plot(fft[:, init_freq_idx].imag)
# plt.xlabel("Time")
# plt.ylabel("Highest amplitude")
# plt.savefig("output/highest_amplitude_1.pdf")
# plt.show()

# for i in range(0, 50, 5):
#     plt.plot(rho_t[i], label=f"time={i}")
# plt.legend()
# plt.show()

# roadmap point 2

x_range = 100
y_range = 100

rho = np.ones(shape=(x_range, y_range))
v_x = epsilon * np.sin(2 * np.pi * np.arange(x_range) / x_range)
v_x = np.einsum('x, y -> xy', v_x, np.ones(y_range))
v_max = np.max(np.abs(v_x))
v_pos = np.abs(np.where(v_x > 0, v_x, 0))
v_neg = np.abs(np.where(v_x < 0, v_x, 0))
v_neu = (np.ones_like(v_x) * v_max) - (v_pos + v_neg)
f_2 = np.zeros(shape=(9, x_range, y_range))
f_2[1] = v_neg
f_2[3] = v_pos
f_2[0] = v_neu

# # single simulation
# sim_2 = Simulation(f=f_2, omega=0.6, t_range=1000)
# v_x_t = sim_2.calc_v_t()[0, :, :, 0]
# fft = np.fft.fft(v_x_t)
# freq = np.fft.fftfreq(v_x_t.shape[-1])
# init_freq_idx = np.argmax(fft[0].imag)

# # v_x_t over time
# for i in range(0, 1000, 100):
#     plt.plot(v_x_t[i], label=f"time={i}")
# # plt.legend()
# plt.xlabel("y")
# plt.ylabel("$u_x(y)$")
# plt.savefig("output/v_x_t.pdf")
# plt.show()

# # Fast fourier transformation
# fft = np.fft.fft(v_x_t)
# freq = np.fft.fftfreq(v_x_t.shape[-1])
# init_freq_idx = np.argmax(fft[0].imag)
# plt.plot(fft[:, init_freq_idx].imag, label=f"$\\omega$ = {om}")
# plot the highest frequency at every point in time
# plt.plot(fft[np.arange(fft.shape[0]), np.argmax(fft.imag, axis=1)].imag)


# # amplitude over time for different omegas
# omegas = np.linspace(0.03, 0.6, 6)
# viskosities = []
# for om in omegas:
#     sim_2 = Simulation(f=copy(f_2), omega=om, t_range=1000)
#     v_x_t = sim_2.calc_v_t()[0, :, :, 0]
#     amplitude = -v_x_t[:, x_range // 4]
#     viskosities.append(np.size(amplitude) / np.sum(amplitude))
#     plt.plot(amplitude, label=f"$\\omega$ = {np.round(om, 2)}")
# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("Highest amplitude")
# plt.savefig("output/highest_amplitude.pdf")
# plt.show()


# viskosities for different omegas
omegas = np.linspace(0.03, 0.6, 20)
viskosities = []
amplitudes = []
t_range = 1500
for om in omegas:
    sim_2 = Simulation(f=copy(f_2), omega=om, t_range=t_range)
    v_x_t = sim_2.calc_v_t()[0, :, :, 0]
    amplitude = -v_x_t[:, x_range // 4]
    viskosities.append(np.polyfit(np.arange(t_range), -np.log(amplitude), 1)[0])
    amplitudes.append(amplitude)
plt.plot(omegas, np.array(viskosities) * (y_range / (2 * np.pi))**2, label="Simulated viskosities")
omega_range = np.linspace(0.03, 0.6, 200)
plt.plot(omegas, 1 / 3 * (1 / omegas - 1 / 2), label="Analytical prediction")
plt.legend()
plt.xlabel("Damping frequency $\\omega$")
plt.ylabel("Viskosity $\\eta$")
plt.savefig("./output/viskosities_over_omega.pdf")
plt.show()

# vis = []
# for amplitude in amplitudes:
#     vis.append(np.polyfit(np.arange(t_range), - np.log(amplitude), 1)[0])
# plt.plot(omegas, np.array(vis) * (y_range / np.pi)**2, label="Simulated viskosities")
# omega_range = np.linspace(0.03, 0.6, 200)
# plt.plot(omegas, 1 / 3 * (1 / omegas - 1 / 2), label="Analytical prediction")
# plt.legend()

plt.show()

# sim_2.animate_f_t(interval=50, save='output/shear_wave_f_2.gif')
# sim_2.animate_rho_t(interval=50, save="output/shear_wave_rho2.gif")
