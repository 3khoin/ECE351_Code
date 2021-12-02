# imported packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy import fftpack
from scipy import signal

fs = 1e6

# load input signal
df = pd.read_csv('NoisySignal.csv')

t = df['0'].values
sensor_sig = df['1'].values

plt.figure(figsize = (10, 7))
plt.plot(t, sensor_sig)
plt.grid()
plt.title('Noisy Input Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.show()

def fft(x):
    N = len(x)
    X_fft = scipy.fftpack.fft(x)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft)
    
    freq = np.arange(-N/2, N/2) * fs/N
    
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.zeros(len(X_mag))
    # New
    for i in range(len(X_mag)):
        if( X_mag[i] > 1e-8 ):
            X_phi[i] = np.angle(X_fft_shifted[i])
    return freq, X_mag, X_phi

def make_stem(ax, x, y, color='k', style='solid', label='', linewidths=2.5, **kwargs):
    ax.axhline(x[0], x[-1], 0, color='r')
    ax.vlines(x, 0, y, color=color, linestyles=style, label=label, linewidths=linewidths)
    ax.set_ylim([1.05*y.min(), 1.05*y.max()])
    
def dB(x):
    y = 20 * np.log10(x)
    return y

freq, X_mag, X_phi = fft(sensor_sig)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10,7))

plt.subplot(ax1)
make_stem(ax1, freq, X_mag)
plt.grid()
plt.ylabel('Magnitude of X(s)')

# 60 Hz, Magnitude ~ .75

plt.subplot(ax2)
plt.xlim(1750, 2050)
make_stem(ax2, freq, X_mag)
plt.grid()

plt.subplot(ax3)
plt.xlim(0, 1700)
make_stem(ax3, freq, X_mag)
plt.grid()

plt.subplot(ax4)
plt.xlim(49750, 51250)
make_stem(ax4, freq, X_mag)
plt.xlabel('Frequency [Hz]')
plt.grid()

# =============================================================================
# plt.subplot(ax3)
# make_stem(ax3, freq, X_phi)
# plt.grid()
# plt.ylabel('Phase of X(s)')
# 
# plt.subplot(ax4)
# plt.xlim(-150, 150)
# make_stem(ax4, freq, X_phi)
# plt.grid()
# plt.xlabel('f [Hz]')
# =============================================================================
