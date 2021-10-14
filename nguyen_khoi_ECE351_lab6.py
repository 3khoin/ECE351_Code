# Imported packages
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal

steps = 1e-3 # step size
xmin = 0 # min x on plot
xmax = 2 # max x on plot
t = np.arange(xmin, xmax + steps, steps)

def u(t): # Step function definition
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] > 0:
            y[i] = 1
    return y

def y(t):
    y = (1/2) * (1 - np.exp(-4 * t) + 2 * np.exp(-6 * t)) * u(t)
    return y

def cos_method(roots, poles, t):
    y = 0
    for i in range(len(poles)):
        mag = np.absolute(roots[i])
        ang = np.angle(roots[i])
        alpha = np.real(poles[i])
        omega = np.imag(poles[i])
        y += mag * np.exp(alpha * t) * np.cos(omega * t + ang) * u(t)
    return y

# Plots
plt.figure(figsize = (15, 10))
plt.subplot(3, 1, 1)
plt.plot(t, y(t))
plt.grid()
plt.title('Part 1, Task 1')
plt.ylabel('y(t) (hand-calculated)')

Hnum = [1, 6, 12]
Hden = [1, 10, 24]

ts = np.arange(xmin, xmax + steps, steps)
ts, Hs = scipy.signal.step((Hnum, Hden), T = ts)

plt.figure(figsize = (15, 10))
plt.subplot(3, 2, 1)
plt.plot(ts, Hs)
plt.grid()
plt.title('Part 1, Task 2')
plt.ylabel('y(t) (scipy.signal.step()')

Hden1 = [1, 10, 24, 0]

[roots, poles, _] = scipy.signal.residue(Hnum, Hden1)
print('Part 1 Task 3:\n', roots, poles)

Hnum2 = [25250]
Hden2 = [1, 18, 218, 2036, 9085, 25250, 0]

ts2 = np.arange(xmin, 4.5 + steps, steps)
[roots2, poles2, _] = scipy.signal.residue(Hnum2, Hden2)
print('Part 2 Task 1:\n', roots2, poles2)

Hs2 = cos_method(roots2, poles2, ts2)
plt.figure(figsize = (15, 10))
plt.subplot(3, 3, 1)
plt.plot(ts2, Hs2)
plt.grid()
plt.title('Part 2, Task 2')
plt.ylabel('y(t) (scipy.signal.step()')

Hden2_2 = [1, 18, 218, 2036, 9085, 25250]
ts2, Hs2_2 = scipy.signal.step((Hnum2, Hden2_2), T = ts2)
plt.figure(figsize = (15, 10))
plt.subplot(3, 4, 1)
plt.plot(ts2, Hs2_2)
plt.grid()
plt.title('Part 2, Task 3')
plt.ylabel('y(t) (scipy.signal.step()')