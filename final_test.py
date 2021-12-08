# Imported packages
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal as sig
import control as con

R = 1000
L = 35.1e-3
C = 200e-9

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

steps = 50 # step size
xmin = 1e1 # min x on plot
xmax = 1e6 # max x on plot
omega = np.arange(xmin, xmax + steps, steps)

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
plt.title('Part 1 Task 3')

# UP CLOSE
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
plt.title('UP CLOSE')

# UP CLOSE
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
plt.title('UP CLOSE')

# UP CLOSE
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
plt.title('UP CLOSE')