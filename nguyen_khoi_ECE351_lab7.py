# Imported packages
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal

steps = 1e-3 # step size
xmin = 0 # min x on plot
xmax = 10 # max x on plot
t = np.arange(xmin, xmax + steps, steps)

Gnum = [1,9]
Gdem = scipy.signal.convolve([1, -6, -16], [1, 4])
Anum = [1, 4]
Adem = [1, 4, 3]
B = [1, 26, 168]

[Gzeros, Gpoles, _] = scipy.signal.tf2zpk(Gnum, Gdem)
[Azeros, Apoles, _] = scipy.signal.tf2zpk(Anum, Adem)
Bzeros = np.roots(B)

print("G(s) zeros:", Gzeros, "poles:", Gpoles)
print("A(s) zeros:", Azeros, "poles:", Apoles)
print("B(s) zeros:", Bzeros)
print("")

openloopnum = scipy.signal.convolve(Gnum, Anum)
openloopdem = scipy.signal.convolve(Gdem, Adem)

t, openloop = scipy.signal.step((openloopnum, openloopdem), T = t)

plt.figure(figsize = (15, 10))
plt.subplot(3, 1, 1)
plt.plot(t, openloop)
plt.grid()
plt.title('Part 1, Task 5')
plt.ylabel('Open-loop step response')

print("Open-loop transfer function")
print("Numerator: ", np.roots(openloopnum))
print("Denominator: ", np.roots(openloopdem))
print("")

closedloopnum = scipy.signal.convolve(Gnum, Anum)
closedloopdem1 = scipy.signal.convolve(Adem, Gdem)
closedloopdem2_1 = scipy.signal.convolve(Adem, Gnum)
closedloopdem2 = scipy.signal.convolve(B, closedloopdem2_1)
closedloopdem = closedloopdem1 + closedloopdem2

t, closedloop = scipy.signal.step((closedloopnum, closedloopdem), T = t)

plt.figure(figsize = (15, 10))
plt.subplot(3, 1, 1)
plt.plot(t, closedloop)
plt.grid()
plt.title('Part 2, Task 4')
plt.ylabel('Closed-loop step response')

print("Closed-loop transfer function")
print("Numerator: ", closedloopnum)
print("Denominator: ", closedloopdem)