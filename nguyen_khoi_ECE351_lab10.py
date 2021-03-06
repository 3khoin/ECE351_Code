# Imported packages
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal as sig
import control as con

R = 1000
L = 27e-3
C = 100e-9

def H_mag(w):
    num = w/(R * C)
    dem = np.sqrt((1/(L * C) - w**2)**2 + (w/(R * C))**2)
    y = num/dem
    return y

def H_phase(w):
    y = np.zeros(len(w))
    for i in range(len(w)):
        add = np.arctan(w[i]/(R * C)) - np.arctan((w[i]/(R * C))/(1/(L * C) - (w[i])**2))
        if( add > 90 * np.pi/180 ):
           y[i] += add - np.pi
        else:
           y[i] += add
    return y

def dB(x):
    y = 20 * np.log10(x)
    return y

def x(t):
    y = np.cos(2*np.pi*100*t) + np.cos(2*np.pi*3024*t) + np.sin(2*np.pi*50000*t)
    return y

steps = 100 # step size
xmin = 1e3 # min x on plot
xmax = 1e6 # max x on plot
omega = np.arange(xmin, xmax + steps, steps)

num = [1/(R * C), 0]
den = [1, 1/(R * C), 1/(L * C)]

H = (num, den)
w, mag, phase = sig.bode(H, omega)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.semilogx(omega, dB(H_mag(omega)))
plt.grid()
plt.title('Part 1 Task 1')
plt.ylabel('Magnitude (dB)')

plt.subplot(3, 1, 2)
plt.semilogx(w, H_phase(w) * 180/np.pi)
plt.grid()
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Phase (degrees)')

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.semilogx(w, mag)
plt.grid()
plt.title('Part 1 Task 2')
plt.ylabel('Magnitude (dB)')

plt.subplot(3, 1, 2)
plt.semilogx(w, phase)
plt.grid()
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Phase (degrees)')

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.semilogx(w, phase)
plt.grid()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (degrees)')

sys = con.TransferFunction(num, den)
_ = con.bode(sys, omega, dB=True, Hz=True, deg=True, plot=True)
plt.title('Part 1 Task 3')

fs = 1e6

steps = 1/fs # step size
xmin = 0 # min x on plot
xmax = 1e-2 # max x on plot
t = np.arange(xmin, xmax, steps)

filternum, filterdem = sig.bilinear(num, den, fs)
y = sig.lfilter(filternum, filterdem, x(t))

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t, x(t))
plt.grid()
plt.title("Part 2 Task 1")
plt.xlabel("Time (s)")
plt.ylabel("x(t)")

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t, y)
plt.grid()
plt.title("Part 2 Task 4")
plt.xlabel("Time (s)")
plt.ylabel("y(t)")