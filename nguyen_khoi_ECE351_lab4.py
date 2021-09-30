# Imported packages
import numpy as np
import matplotlib.pyplot as plt

def u(t): # Step function definition
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] > 0:
            y[i] = 1
    return y

# Convolution definition
def conv(f1, f2):
    f1new = np.append(f1, np.zeros((1, len(f2)-1)))
    f2new = np.append(f2, np.zeros((1, len(f1)-1)))
    result = np.zeros(f1new.shape)
    
    for i in range(len(f1) + len(f2) - 2):
        result[i] = 0
        for j in range(len(f1)):
            if(i - j + 1 > 0):
                try:
                    result[i] += f1new[j] * f2new[i-j+1]
                except:
                    print(i,j)
    return result

def h1(t):
    y = np.exp(-2*t) * (u(t) - u(t-3))
    return y

def h2(t):
    y = u(t-2) - u(t-6)
    return y

def h3(t): # w_0 = f_0 * 2 * pi, f_0 = 0.25 Hz
    y = np.cos((0.25*(2 * np.pi)) * t) * u(t)
    return y

steps = 1e-2 # step size
xmin = -10 # min x on plot
xmax = 10 # max x on plot
t = np.arange(xmin, xmax + steps, steps)

# Plots for h1, h2, h3
plt.figure(figsize = (15, 10))
plt.subplot(3, 1, 1)
plt.plot(t, h1(t))
plt.grid()
plt.title('Part 1, Task 1 functions')
plt.ylabel('h1(t)')

plt.subplot(3, 1, 2)
plt.plot(t, h2(t))
plt.grid()
plt.ylabel('h2(t)')

plt.subplot(3, 1, 3)
plt.plot(t, h3(t))
plt.grid()
plt.ylabel('h3(t)')
plt.show()

####

h1c = conv(h1(t), u(t)) * steps
h2c = conv(h2(t), u(t)) * steps
h3c = conv(h3(t), u(t)) * steps

t2 = np.arange(2*t[0], 2*t[len(t)-1] + steps, steps)

# Plots for the step responses of h1, h2, h3
plt.figure(figsize = (15, 10))
plt.xlim([-10, 10])
plt.subplot(3, 2, 1)
plt.plot(t2, h1c)
plt.grid()
plt.title('Part 2, Task 1 step responses')
plt.ylabel('h1(t) * u(t)')
plt.xlim([-10, 10])

plt.subplot(3, 2, 2)
plt.plot(t2, h2c)
plt.grid()
plt.ylabel('h2(t) * u(t)')
plt.xlim([-10, 10])

plt.subplot(3, 2, 3)
plt.plot(t2, h3c)
plt.grid()
plt.ylabel('h3(t) * u(t)')
plt.xlim([-10, 10])
plt.show()

h1s = (1/2) * ( ( (1 - np.exp(-2 * t)) * u(t)) - (np.exp(-6) * (1 - np.exp(-2 * (t-3))) * u(t - 3)) )
# h1s = (1/2) * ((1 - np.exp(-2 * t2)) * u(t2) - (1 - np.exp(-2 * (t2 - 3))) * u(t2-3))
h2s = ((t - 2) * u(t - 2)) - ((t - 6) * u(t - 6))
h3s = 1/(0.25 * 2 * np.pi) * np.sin(0.25 * 2 * np.pi * t) * u(t)

# Plots for the hand-calculated step responses of h1, h2, h3
plt.figure(figsize = (15, 10))
plt.subplot(3, 3, 1)
plt.plot(t, h1s)
plt.grid()
plt.title('Part 2, Task 2 step responses')
plt.ylabel('h1(t) * u(t)')

plt.subplot(3, 3, 2)
plt.plot(t, h2s)
plt.grid()
plt.ylabel('h2(t) * u(t)')

plt.subplot(3, 3, 3)
plt.plot(t, h3s)
plt.grid()
plt.ylabel('h3(t) * u(t)')
plt.show()