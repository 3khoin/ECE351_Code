# imported packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy import fftpack
import scipy.signal as sig
import control as con

fs = 1e6 # Sampling frequency

# Component values
R = 1000
L = 35.1e-3
C = 200e-9

def fft(x): # Fast Fourier Transform
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

# Faster stem plot
def make_stem(ax, x, y, color='k', style='solid', label='', linewidths=2.5, **kwargs):
    ax.axhline(x[0], x[-1], 0, color='r')
    ax.vlines(x, 0, y, color=color, linestyles=style, label=label, linewidths=linewidths)
    ax.set_ylim([1.05*y.min(), 1.05*y.max()])
    
# Converts a value to dB
def dB(x):
    y = 20 * np.log10(x)
    return y

# load input signal
df = pd.read_csv('NoisySignal.csv')

# Initialize input signal from .csv file
t = df['0'].values
sensor_sig = df['1'].values
sensor_sig2 = df['1'].values

# Unfiltered signal plot
plt.figure(figsize = (10, 7))
plt.plot(t, sensor_sig)
plt.grid()
plt.title('Noisy Input Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.show()

freq, X_mag, X_phi = fft(sensor_sig)

# FFT plots of unfiltered signal
fig, ax = plt.subplots(figsize=(10,7))

make_stem(ax, freq, X_mag)
plt.grid()
plt.ylabel('|X(f)|')
plt.xlabel('Frequency [Hz]')
plt.title('Noisy Input Signal')
plt.show()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10,7))

plt.subplot(ax1)
plt.xlim(1750, 2050)
make_stem(ax1, freq, X_mag)
plt.grid()
plt.ylabel('|X(f)|')

plt.subplot(ax2)
plt.xlim(0, 1700)
make_stem(ax2, freq, X_mag)
plt.grid()
plt.ylabel('|X(f)|')

plt.subplot(ax3)
plt.xlim(49750, 51250)
make_stem(ax3, freq, X_mag)
plt.ylabel('|X(f)|')
plt.grid()

plt.subplot(ax4)
plt.xlim(100000, 400000)
make_stem(ax4, freq, X_mag)
plt.ylabel('|X(f)|')
plt.xlabel('Frequency [Hz]')
plt.grid()

# Bode plots
# Full range view of the filter
steps = 100 # step size
xmin = 1e1 # min x on plot
xmax = 1e6 # max x on plot
omega = np.arange(xmin, xmax + steps, steps)

# Numerator and denominator of H(s)
num = [1/(R * C), 0]
den = [1, 1/(R * C), 1/(L * C)]

H = (num, den)
w, mag, phase = sig.bode(H, omega)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.semilogx(w, phase)
plt.grid()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (degrees)')

sys = con.TransferFunction(num, den)
_ = con.bode(sys, omega, dB=True, Hz=True, deg=True, plot=True)
plt.title('Filter Bode Plot')

# 1800 to 2000 Hz
xmin = 1800*2*np.pi # min x on plot
xmax = 2000*2*np.pi # max x on plot
omega2 = np.arange(xmin, xmax + steps, steps)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.semilogx(w, phase)
plt.grid()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (degrees)')

sys = con.TransferFunction(num, den)
_ = con.bode(sys, omega2, dB=True, Hz=True, deg=True, plot=True)
plt.title('Filter Bode (1800 to 2000 Hz)')

# 50 to 70 Hz
xmin = 50*2*np.pi # min x on plot
xmax = 70*2*np.pi # max x on plot
omega3 = np.arange(xmin, xmax + steps, steps)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.semilogx(w, phase)
plt.grid()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (degrees)')

sys = con.TransferFunction(num, den)
_ = con.bode(sys, omega3, dB=True, Hz=True, deg=True, plot=True)
plt.title('Filter Bode (50 to 70 Hz)')

# 49000 to 51000 Hz
xmin = 49000*2*np.pi # min x on plot
xmax = 51000*2*np.pi # max x on plot
omega4 = np.arange(xmin, xmax + steps, steps)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.semilogx(w, phase)
plt.grid()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (degrees)')

sys = con.TransferFunction(num, den)
_ = con.bode(sys, omega4, dB=True, Hz=True, deg=True, plot=True)
plt.title('Filter Bode (49000 to 51000 Hz)')

# 100000 to 400000 Hz
xmin = 100000*2*np.pi # min x on plot
xmax = 400000*2*np.pi # max x on plot
omega4 = np.arange(xmin, xmax + steps, steps)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.semilogx(w, phase)
plt.grid()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (degrees)')

sys = con.TransferFunction(num, den)
_ = con.bode(sys, omega4, dB=True, Hz=True, deg=True, plot=True)
plt.title('Filter Bode (100000 to 400000 Hz)')

# Filtered signal
steps2 = 1/fs # step size
xmin2 = 0 # min x on plot
xmax2 = 1e-2 # max x on plot
t2 = np.arange(xmin2, xmax2, steps2/5 )

filternum, filterdem = sig.bilinear(num, den, fs)
y2 = sig.lfilter(filternum, filterdem, sensor_sig2)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t2, y2)
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.title("Filtered Signal")

freq2, X_mag2, X_phi2 = fft(y2)

# FFT plots of filtered signal
fig, ax = plt.subplots(figsize=(10,7))

make_stem(ax, freq2, X_mag2)
plt.grid()
plt.ylabel('|X(f)|')
plt.xlabel('Frequency [Hz]')
plt.title('Noisy Input Signal')
plt.show()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10,7))

plt.subplot(ax1)
plt.xlim(1750, 2050)
make_stem(ax1, freq, X_mag2)
plt.grid()
plt.ylabel('|X(f)|')

plt.subplot(ax2)
plt.xlim(0, 1700)
make_stem(ax2, freq, X_mag2)
plt.grid()
plt.ylabel('|X(f)|')

plt.subplot(ax3)
plt.xlim(49750, 51250)
make_stem(ax3, freq, X_mag2)
plt.ylabel('|X(f)|')
plt.xlabel('Frequency [Hz]')
plt.grid()

plt.subplot(ax4)
plt.xlim(100000, 400000)
make_stem(ax4, freq, X_mag2)
plt.grid()
plt.ylabel('|X(f)|')