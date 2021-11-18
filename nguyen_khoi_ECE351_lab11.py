# Imported packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import scipy
import scipy.signal as sig

def zplane(b, a, filename = None):
    ax = plt.subplot(1, 1, 1)
    
    uc = patches.Circle((0,0), radius=1, fill=False, color='black', ls='dashed')
    ax.add_patch(uc)
    
    if np.max(b) > 1:
        kn = np.max(b)
        b = np.array(b)/float(kn)
    else:
        kn = 1
        
    if np.max(a) > 1:
        kd = np.max(a)
        a = np.array(a)/float(kd)
    else:
        kd = 1
    
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    
    t1 = plt.plot(z.real, z.imag, 'o', ms=10, label='Zeros')
    plt.setp(t1, markersize=10.0, markeredgewidth=1.0)
    
    t2 = plt.plot(p.real, p.imag, 'x', ms=10, label='Poles')
    plt.setp(t2, markersize=12.0, markeredgewidth=3.0)
    
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.legend()
    
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        
    return z, p, k

num = [2, -40]
den = [1, -10, 16]
r, p, k = sig.residuez(num, den)

zplane(num, den)
w, h = sig.freqz(num, den, whole=True)

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(w, 20 * np.log10(abs(h)), 'b')
plt.grid()
plt.title('Task 5')
plt.xlabel('Frequency (rads)')
plt.ylabel('Magnitude (dB)')

plt.subplot(3, 1, 2)
plt.plot(w, np.angle(h) * 180/np.pi, 'g')
plt.grid()
plt.xlabel('Frequency (rads)')
plt.ylabel('Phase')

print("r:", r)
print("p:", p)
print("k:", k)