from numpy import sin, cos, pi, arange
from numpy.random import randint
import matplotlib.pyplot as plt
import pandas as pd

fs = 1e6
Ts = 1/fs
t_end = 50e-3

t = arange(0,t_end-Ts,Ts)

f1 = 1.8e3
f2 = 1.9e3
f3 = 2e3
f4 = 1.85e3
f5 = 1.87e3
f6 = 1.94e3
f7 = 1.92e3

info_signal = (2.5*cos(2*pi*f1*t) + 1.75*cos(2*pi*f2*t) + 2*cos(2*pi*f3*t) + 
               2*cos(2*pi*f4*t) + 1*cos(2*pi*f5*t) + 1*cos(2*pi*f6*t) +
               1.5*cos(2*pi*f7*t))

N = 25
my_sum = 0

for i in range(N+1):
    noise_amp     = 0.075*randint(-10,10,size=(1,1))
    noise_freq    = randint(-1e6,1e6,size=(1,1))
    noise_signal  = my_sum + noise_amp * cos(2*pi*noise_freq*t)
    my_sum = noise_signal

f6 = 50e3    #50kHz
f7 = 49.9e3
f8 = 51e3

pwr_supply_noise = 1.5*sin(2*pi*f6*t) + 1.25*sin(2*pi*f7*t) + 1*sin(2*pi*f8*t)

f9 = 60

low_freq_noise = 1.5*sin(2*pi*f9*t)

total_signal = info_signal + noise_signal + pwr_supply_noise + low_freq_noise
total_signal = total_signal.reshape(total_signal.size)

# plt.figure(figsize=(12,8))
# plt.subplot(3,1,1)
# plt.plot(t,info_signal)
# plt.grid(True)
# plt.subplot(3,1,2)
# plt.plot(t,info_signal+pwr_supply_noise)
# plt.grid(True)
# plt.subplot(3,1,3)
# plt.plot(t,total_signal)
# plt.grid(True)
# plt.show()

df = pd.DataFrame({'0':t,
                   '1':total_signal})

df.to_csv('NoisySignal.csv')