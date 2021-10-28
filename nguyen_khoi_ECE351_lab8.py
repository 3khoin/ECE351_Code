# Imported packages
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal

steps = 1e-3 # step size
xmin = 0 # min x on plot
xmax = 20 # max x on plot
t = np.arange(xmin, xmax + steps, steps)

T = 8

def b(k):
    y = 4/(k * np.pi) * (np.sin(k * np.pi / 2))**2
    return y

def x(t, N):
    y = 0
    for i in range(1, N + 1):
        y += b(i) * np.sin(i * 2 * np.pi * t/ T)
    return y

print('b(1):', b(1))
print('b(2):', b(2))
print('b(3):', b(3))

plt.figure(figsize = (15, 10))
plt.subplot(3, 1, 1)
plt.plot(t, x(t, 1))
plt.grid()
plt.title('Task 2')
plt.ylabel('N = 1')

plt.subplot(3, 1, 2)
plt.plot(t, x(t, 3))
plt.grid()
plt.ylabel('N = 3')

plt.subplot(3, 1, 3)
plt.plot(t, x(t, 15))
plt.grid()
plt.ylabel('N = 15')
plt.show()

plt.figure(figsize = (15, 10))
plt.subplot(3, 1, 1)
plt.plot(t, x(t, 50))
plt.grid()
plt.title('Task 2')
plt.ylabel('N = 50')

plt.subplot(3, 1, 2)
plt.plot(t, x(t, 150))
plt.grid()
plt.ylabel('N = 150')

plt.subplot(3, 1, 3)
plt.plot(t, x(t, 1500))
plt.grid()
plt.ylabel('N = 1500')
plt.show()