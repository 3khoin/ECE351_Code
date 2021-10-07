# Imported packages
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal

def u(t): # Step function definition
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] > 0:
            y[i] = 1
    return y


def h(t):
    y = 0.001036 * np.exp(-5000*t) * np.sin(18585*t + 105 * np.pi/180) * u(t)
    return y

steps = 1e-6 # step size
xmin = 0 # min x on plot
xmax = 1.2e-3 # max x on plot
t = np.arange(xmin, xmax + steps, steps)

R = 1000
L = 27e-3
C = 100e-9

Hnum = [1/(R * C), 0]
Hden = [1, 1/(R * C), 1/(L * C)]

t, H = scipy.signal.impulse((Hnum, Hden), T = t)

# Plots
plt.figure(figsize = (15, 10))
plt.subplot(3, 1, 1)
plt.plot(t, h(t))
plt.grid()
plt.title('Part 1, Task 1 function')
plt.ylabel('h(t) (hand-calculated)')

plt.figure(figsize = (15, 10))
plt.subplot(3, 2, 1)
plt.plot(t, H * steps)
plt.grid()
plt.title('Part 1, Task 2 function')
plt.ylabel('h(t) (scipy.signal.impulse()')

ts = np.arange(xmin, xmax + steps, steps)
ts, Hs = scipy.signal.step((Hnum, Hden), T = ts)

plt.figure(figsize = (15, 10))
plt.subplot(3, 3, 1)
plt.plot(ts, Hs * steps)
plt.grid()
plt.title('Part 2, Task 1 function')
plt.ylabel('H(s) step response')