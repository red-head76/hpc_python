import matplotlib.pyplot as plt
import numpy as np

t = np.arange(256)
sp = np.fft.fft(np.sin(2 * np.pi * t / 256))
freq = np.fft.fftfreq(t.shape[-1])
plt.plot(freq, sp.imag, label="imag")
plt.plot(freq, sp.real, label="real")
plt.legend()
plt.show()
