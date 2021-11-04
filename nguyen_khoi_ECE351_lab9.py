# Imported packages
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import fftpack
from scipy import signal

fs = 100 # Sampling frequency
T = 8

steps = 1/fs # step size
xmin = 0 # min x on plot
xmax = 2 # max x on plot
t = np.arange(xmin, xmax, steps)

def fft(x):
    N = len(x)
    X_fft = scipy.fftpack.fft(x)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft)
    
    freq = np.arange(-N/2, N/2) * fs/N
    
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    return freq, X_mag, X_phi

def fft_new(x):
    N = len(x)
    X_fft = scipy.fftpack.fft(x)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft)
    
    freq = np.arange(-N/2, N/2) * fs/N
    
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.zeros(len(X_mag))
    # New
    for i in range(len(X_mag)):
        if( X_mag[i] > 1e-10 ):
            X_phi[i] = np.angle(X_fft_shifted[i])
    return freq, X_mag, X_phi

# b_k for the Fourier series approximation of the square wave
def b(k):
    y = 4/(k * np.pi) * (np.sin(k * np.pi / 2))**2
    return y

# Fourier series approximation of the square wave
def x(t, N):
    y = 0
    for k in range(1, N + 1):
        y += b(k) * np.sin(k * 2 * np.pi * t/ T)
    return y

# Task 1
func1 = np.cos(2 * np.pi * t)
freq, X_mag, X_phi = fft(func1)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t, func1)
plt.grid()
plt.title('Task 1')
plt.xlabel('t [s]')
plt.ylabel('x(t)')

plt.subplot(3, 2, 3)
plt.stem(freq, X_mag)
plt.grid()
plt.ylabel('Magnitude of X(s)')

plt.subplot(3, 2, 4)
plt.xlim(-2.0, 2.0)
plt.stem(freq, X_mag)
plt.grid()

plt.subplot(3, 2, 5)
plt.stem(freq, X_phi)
plt.grid()
plt.ylabel('Phase of X(s)')
plt.xlabel('f [Hz]')

plt.subplot(3, 2, 6)
plt.xlim(-2.0, 2.0)
plt.stem(freq, X_phi)
plt.grid()
plt.xlabel('f [Hz]')

# Task 2
func2 = 5 * np.sin(2 * np.pi * t)
freq, X_mag, X_phi = fft(func2)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t, func2)
plt.grid()
plt.title('Task 2')
plt.xlabel('t [s]')
plt.ylabel('x(t)')

plt.subplot(3, 2, 3)
plt.stem(freq, X_mag)
plt.grid()
plt.ylabel('Magnitude of X(s)')

plt.subplot(3, 2, 4)
plt.xlim(-2.0, 2.0)
plt.stem(freq, X_mag)
plt.grid()

plt.subplot(3, 2, 5)
plt.stem(freq, X_phi)
plt.grid()
plt.ylabel('Phase of X(s)')
plt.xlabel('f [Hz]')

plt.subplot(3, 2, 6)
plt.xlim(-2.0, 2.0)
plt.stem(freq, X_phi)
plt.grid()
plt.xlabel('f [Hz]')

# Task 3
func3 = 2 * np.cos((2 * np.pi * 2 * t) - 2) + (np.sin((2 * np.pi * 6 * t) + 3))**2
freq, X_mag, X_phi = fft(func3)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t, func3)
plt.grid()
plt.title('Task 1')
plt.xlabel('t [s]')
plt.ylabel('x(t)')

plt.subplot(3, 2, 3)
plt.stem(freq, X_mag)
plt.grid()
plt.ylabel('Magnitude of X(s)')

plt.subplot(3, 2, 4)
plt.xlim(-15.0, 15.0)
plt.stem(freq, X_mag)
plt.grid()

plt.subplot(3, 2, 5)
plt.stem(freq, X_phi)
plt.grid()
plt.ylabel('Phase of X(s)')
plt.xlabel('f [Hz]')

plt.subplot(3, 2, 6)
plt.xlim(-15.0, 15.0)
plt.stem(freq, X_phi)
plt.grid()
plt.xlabel('f [Hz]')

# Task 4-1
freq, X_mag, X_phi = fft_new(func1)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t, func1)
plt.grid()
plt.title('Task 4-1')
plt.xlabel('t [s]')
plt.ylabel('x(t)')

plt.subplot(3, 2, 3)
plt.stem(freq, X_mag)
plt.grid()
plt.ylabel('Magnitude of X(s)')

plt.subplot(3, 2, 4)
plt.xlim(-2.0, 2.0)
plt.stem(freq, X_mag)
plt.grid()

plt.subplot(3, 2, 5)
plt.stem(freq, X_phi)
plt.grid()
plt.ylabel('Phase of X(s)')
plt.xlabel('f [Hz]')

plt.subplot(3, 2, 6)
plt.xlim(-2.0, 2.0)
plt.stem(freq, X_phi)
plt.grid()
plt.xlabel('f [Hz]')

# Task 4-2
freq, X_mag, X_phi = fft_new(func2)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t, func2)
plt.grid()
plt.title('Task 4-2')
plt.xlabel('t [s]')
plt.ylabel('x(t)')

plt.subplot(3, 2, 3)
plt.stem(freq, X_mag)
plt.grid()
plt.ylabel('Magnitude of X(s)')

plt.subplot(3, 2, 4)
plt.xlim(-2.0, 2.0)
plt.stem(freq, X_mag)
plt.grid()

plt.subplot(3, 2, 5)
plt.stem(freq, X_phi)
plt.grid()
plt.ylabel('Phase of X(s)')
plt.xlabel('f [Hz]')

plt.subplot(3, 2, 6)
plt.xlim(-2.0, 2.0)
plt.stem(freq, X_phi)
plt.grid()
plt.xlabel('f [Hz]')

# Task 4-3
freq, X_mag, X_phi = fft_new(func3)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t, func3)
plt.grid()
plt.title('Task 4-3')
plt.xlabel('t [s]')
plt.ylabel('x(t)')

plt.subplot(3, 2, 3)
plt.stem(freq, X_mag)
plt.grid()
plt.ylabel('Magnitude of X(s)')

plt.subplot(3, 2, 4)
plt.xlim(-15.0, 15.0)
plt.stem(freq, X_mag)
plt.grid()

plt.subplot(3, 2, 5)
plt.stem(freq, X_phi)
plt.grid()
plt.ylabel('Phase of X(s)')
plt.xlabel('f [Hz]')

plt.subplot(3, 2, 6)
plt.xlim(-15.0, 15.0)
plt.stem(freq, X_phi)
plt.grid()
plt.xlabel('f [Hz]')

# Task 5
t2 = np.arange(0, 16, steps)
func4 = x(t2, 15)
freq, X_mag, X_phi = fft_new(func4)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t2, func4)
plt.grid()
plt.title('Task 5')
plt.xlabel('t [s]')
plt.ylabel('x(t)')

plt.subplot(3, 2, 3)
plt.stem(freq, X_mag)
plt.grid()
plt.ylabel('Magnitude of X(s)')

plt.subplot(3, 2, 4)
plt.xlim(-2.0, 2.0)
plt.stem(freq, X_mag)
plt.grid()

plt.subplot(3, 2, 5)
plt.stem(freq, X_phi)
plt.grid()
plt.ylabel('Phase of X(s)')
plt.xlabel('f [Hz]')

plt.subplot(3, 2, 6)
plt.xlim(-2.0, 2.0)
plt.stem(freq, X_phi)
plt.grid()
plt.xlabel('f [Hz]')