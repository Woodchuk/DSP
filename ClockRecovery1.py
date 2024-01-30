import numpy as np
import random as rnd
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import time as tm

N = int(15000) # Number of symbols
EsN0 = 10.0     # Symbol energy to noise spectral density ratio (fractional)
Ns = 100       # Number of samples per symbol

Symbols = [1.0+1.0j, 1.0-1.0j, -1.0-1.0j, -1.0+1.0j]
Eps = 1.0e-1
# Generate set of random symbols
D = rnd.choices([0, 1, 2, 3], weights=None, k=N)
Data = np.zeros(N, dtype=np.complex128)
for i in range(N):
    Data[i] = Symbols[D[i]]

# Generate raised cosine pulse
x = np.linspace(0, Ns, num=Ns, endpoint=False)
p = 0.5 * (1.0 - np.cos(2.0 * np.pi * x / Ns))

# Generate noise
sigma = 1.0 / np.sqrt(2.0 * EsN0 * Ns)
n = np.random.default_rng().normal(0.0, sigma, N*Ns)+1j*np.random.default_rng().normal(0.0, sigma, N*Ns)

#interpolate data
t = np.linspace(0, N, N*Ns)
Data_ext = np.zeros(N*Ns, dtype=np.complex128)
for i in range(N*Ns):
    Data_ext[i] = n[i]
for i in range(N):
    Data_ext[i*Ns] += Data[i]
#filter symbols
Tx = np.convolve(Data_ext, p, mode='same')

# Detect symbol transitions and correct timing
t_offset = 0 # initial offset
p_sample = 15 # initial sample position
avg_len = 8 # Number of averages used in Gardner algorithm
corr_step = 1 # Correction step size
Fil = np.zeros(avg_len, dtype=np.complex128)
Err = np.zeros(N, dtype=np.float64)
Offset = np.zeros(N, dtype=np.float64)
Clock = np.zeros(N*Ns, dtype=np.float64)
for i in range(1,N-1):
    Sc = Tx[i*Ns+t_offset+p_sample]  # prompt sample
    Se = Tx[i*Ns+t_offset+p_sample-int(Ns/2)] # early sample
    Sl = Tx[i*Ns+t_offset+p_sample+int(Ns/2)] # late sample
    Fil[i % avg_len] = (Sl - Se) * np.conjugate(Sc) # Loop filter (average)
    Clock[i*Ns+t_offset+p_sample+int(Ns/2)] = 1.0
    corr = np.mean(Fil.real)
    if(corr > Eps):
        t_offset += -1
    elif(corr < -Eps):
        t_offset += 1
    t_offset = t_offset % Ns
    Err[i] = corr
    Offset[i] = t_offset

fig, ax = plt.subplots(2,1,facecolor='sienna')
ps1 = ax[0].plot(t, Tx.real, 'r-', label='Tx real')
ps2 = ax[0].plot(t, Tx.imag, 'b-', label='Tx imag')
ps2 = ax[0].plot(t, Clock, 'g-', label='Tx imag')
pt1 = mp.Patch(color='red', label='Rx real')
pt2 = mp.Patch(color='blue', label='Rx mag')
pt3 = mp.Patch(color='green', label='Sample clock')
l1 = ax[0].legend(handles=[pt1, pt2, pt3], loc='lower right', shadow=True)
ax[0].grid(True)
ax[0].set_ylabel('Signal amplitude')
ax[0].set_title('filtered baseband signal')
ax[0].set_xlim(0,100)
td = np.linspace(0, N, N)
ps1 = ax[1].plot(td, Err, 'r-', label='Tx real')
ps2 = ax[1].plot(td, Offset, 'b-', label='Tx imag')
pt1 = mp.Patch(color='red', label='Error signal')
pt2 = mp.Patch(color='blue', label='Offset')
l1 = ax[1].legend(handles=[pt1, pt2], loc='lower right', shadow=True)
ax[1].grid(True)
ax[1].set_ylabel('Value')
ax[1].set_title('Error and offset')
ax[1].set_xlim(0,100)
plt.show()

sys.exit(0)

