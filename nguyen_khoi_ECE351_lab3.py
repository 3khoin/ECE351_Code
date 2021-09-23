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

def r(t): # Ramp function definition
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] > 0:
            y[i] = t[i]
    return y 

steps = 1e-2 # step size
xmin = 0 # min x on plot
xmax = 20 # max x on plot
t = np.arange(xmin, xmax + steps, steps)

def f1(t):
    y = u(t-2) - u(t-9)
    return y

def f2(t):
    y = np.exp(-t) * u(t)
    return y

def f3(t):
    y = r(t-2)*((u(t-2) - u(t-3)) + (r(4-t) * (u(t-3) - u(t-4))))
    return y

# Plots for f1, f2, f3
plt.figure(figsize = (15, 10))
plt.subplot(3, 1, 1)
plt.plot(t, f1(t))
plt.grid()
plt.ylabel('f1(t)')

plt.subplot(3, 1, 2)
plt.plot(t, f2(t))
plt.grid()
plt.ylabel('f2(t)')

plt.subplot(3, 1, 3)
plt.plot(t, f3(t))
plt.grid()
plt.ylabel('f3(t)')
plt.show()

# Convolution definition
# First, "clones" of the functions passed as arguments are created, with their lengths being extended by the length of the other. This is because the domain of a convolution combines that of its two input functions.
# A result variable, which is to be later returned, is created in the shape of one of the new function clones.
# The double for loop implements the definition of an integral; at each index i in the combined length of the two functions, the second function at a point i-j+1 is appended with the product of its value at that point and the value of all points preceding it from the first function. This supports the definition of a convolution defined as the output function being the "percent overlap" of the two input functions.
# The result array is then returned.
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

y1 = conv(f1(t), f2(t))
y2 = conv(f2(t), f3(t))
y3 = conv(f1(t), f3(t))

t2 = np.arange(xmin, 2*(xmax + steps), steps)

# Convolution plots
plt.figure(figsize = (15, 10))
plt.subplot(3,2,1)
plt.plot(t2, y1)
plt.grid()
plt.title('Convolutions (function implementation)')
plt.ylabel('f1(t) * f2(t)')

plt.subplot(3,2,2)
plt.plot(t2, y2)
plt.grid()
plt.ylabel('f2(t) * f3(t)')

plt.subplot(3,2,3)
plt.plot(t2, y3)
plt.grid()
plt.ylabel('f1(t) * f3(t)')
plt.show()

# Verification
c1 = scipy.signal.convolve(f1(t), f2(t))
c2 = scipy.signal.convolve(f2(t), f3(t))
c3 = scipy.signal.convolve(f1(t), f3(t))

plt.figure(figsize = (15, 10))
plt.subplot(3,2,1)
plt.plot(t2, c1)
plt.grid()
plt.title('Convolutions (scipy.signal.convolve())')
plt.ylabel('f1(t) * f2(t)')

plt.subplot(3,2,2)
plt.plot(t2, c2)
plt.grid()
plt.ylabel('f2(t) * f3(t)')

plt.subplot(3,2,3)
plt.plot(t2, c3)
plt.grid()
plt.ylabel('f1(t) * f3(t)')
plt.show()