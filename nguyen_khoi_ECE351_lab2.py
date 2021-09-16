# Imported packages
import numpy as np
import matplotlib.pyplot as plt

steps = 1e-2 # step size
t = np.arange(0, 10 + steps, steps)

def func1(t):
    y = np.cos(t)
    return y

y = func1(t)

# Plot for cosine function
plt.figure(figsize = (10, 7))
plt.subplot(1, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.title('cos(t)')
plt.show()

def step(t): # Step function definition
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] > 0:
            y[i] = 1
    return y

def ramp(t): # Ramp function definition
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] > 0:
            y[i] = t[i]
    return y 

steps = 1e-3
plotstart = -3
t = np.arange(plotstart, 10 + steps, steps)
y = step(t) # Step function

# Plot for step function
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.title('Step function')
plt.show()

y = 2 * ramp(t) # Ramp function

# Plot for ramp function
plt.subplot(3, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.title('Ramp function')
plt.show() 

def func2(t):
    y = ramp(t) - ramp (t-3) + 5 * step(t-3) - 2 * step(t-6) - 2 * ramp(t-6)
    return y

steps = 1e-3
xmin = -5
xmax = 10
t = np.arange(xmin, xmax + steps, steps)
y = func2(t)

plt.figure(figsize = (10, 30))
plt.subplot(4, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.title('Figure 2 Plot')
plt.show()

t = np.arange(0 - xmax, 0 - xmin + steps, steps)
y = func2(-t)

plt.figure(figsize = (10, 30))
plt.subplot(5, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.title('Time reversal y(-t)')
plt.show()

t = np.arange(xmin + 4, xmax + 4 + steps, steps)
y = func2(t-4)
y = ramp(t-4) - ramp (t-4-3) + 5 * step(t-4-3) - 2 * step(t-4-6) - 2 * ramp(t-4-6)

plt.figure(figsize = (10, 30))
plt.subplot(6, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.title('Time-shift y(t-4)')
plt.show()

t = np.arange(0 - xmax - 4, 0 - xmin - 4 + steps, steps)
y = func2(-t-4)

plt.figure(figsize = (10, 30))
plt.subplot(7, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.title('Time-shift -y(-t-4)')
plt.show()

t = np.arange(2 * xmin, 2 * xmax + steps, steps)
y = func2(t/2)

plt.figure(figsize = (10, 30))
plt.subplot(8, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.title('Time-scale y(t/2)')
plt.show()

t = np.arange(1/2 * xmin, 1/2 * xmax + steps, steps)
y = func2(2*t)

plt.figure(figsize = (10, 30))
plt.subplot(9, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.title('Time-scale y(2t)')
plt.show()

t = np.arange(xmin, xmax+1, 1)
y = y = ramp(t) - ramp (t-3) + 5 * step(t-3) - 2 * step(t-6) - 2 * ramp(t-6)
z = np.diff(func2(t-1))
t = np.arange(xmin, xmax, 1)

plt.figure(figsize = (10, 30))
plt.subplot(4, 1, 1)
plt.plot(t, z)
plt.grid()
plt.ylabel('y(t)')
plt.title('Differentiation of t')
plt.show()
