import numpy as np
import random as rnd
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import time as tm

# Set up QPSK link with AGWN: carrier, clock recovery schemes and FEC (using Voyager code) as a simulation

Nrep = 8 # number of times to repeat preamble
f_off = 0.1 #carrier offset frequency
R = 1.0 # Symbol rate
Bw = 8.0 # Channel BW
EsN0 = 5 # Symbol energy / Noise spectral density ratio
# Freq correction params
Kp = 0.000182
Ki = 0.000005
G = 1

Ambig = 0.0+0j
Ifil = 0+0j
phi0 = 0.0
Eps = 1e-3


# rate 1/2 length 7 encoder weights (Voyager convolutional code)
weight1 = 0x4f # 1001111
weight2 = 0x6d # 1101101
state = int(0)

N_data = 8192 # number of data bits in packet
N_pre = 64 # number of preamble symbols
Ns = 16 # Number of samples per symbol
Npad = 20 # pad with 0s at the end of frame
Symbols = [[1.0, 1.0, 0], [1.0, -1.0, 1], [-1.0, -1.0, 2],[-1.0, 1.0, 3]]

def Encoder(x):
   global state
   state <<= 1
   state += (x & 0x1)
   state &= 0x7f
   OutI = weight1 & state
   OutQ = weight2 & state
   m = Parity(OutI)
   n = Parity(OutQ)
   sym = n + (m << 1)

   return sym # 00 = 0, 01 = 1, 10 = 2, 11 = 3

# Viterbi decoder for 1/2, length 7 code   
def Decoder(Sf, N, Ns, sigma):
   N_sym = int(N / Ns)
   V = np.zeros((N_sym, 128), dtype=np.float64)
   Bs = np.zeros((N_sym, 128), dtype=np.int64)
   Path = np.zeros(N_sym+1, dtype=np.int64)
   for i in range(128):
      V[0][i] = -1.0e10 
      Bs[0][i] = i
   V[0][0] = 0.0 # First decoder state is 0000000
   
   for t in range(1,N_sym):  # Step thru received symbols
      for i in range(128):   # Search each new possible state
         Pmax = -1.0e10
         for j in [int(i/2), int((i+128)/2)]: # Check each originating state (only two possible)
            ii = Parity(weight1 & (((j<<1)&0x7f)+(i%2))) # Generate the symbols associated with
            qq = Parity(weight2 & (((j<<1)&0x7f)+(i%2))) # the encoder state transition
            sym = qq + (ii << 1)
            Symb = np.complex(Symbols[sym][0], Symbols[sym][1])
            Q = -np.real((Sf[t*Ns] - Symb) * np.conjugate(Sf[t*Ns] - Symb) / (2.0 * sigma * sigma))
            p = V[t-1][j] + np.log(0.5) + Q  # Compute the branch probability for transition
            if p > Pmax:
               Pmax = p
               Smax = j
         V[t][i] = Pmax
         Bs[t][i] = Smax
   Path[N_sym-1] = Smax
   for t in range(N_sym-2, 0, -1):
      Path[t] = Bs[t+1][Path[t+1]]
      
   return Path
   
def Parity(x):
   p = 0
   for i in range(8):
      if(((x & 0xff) >> i) & 0x1 == 1):
         p ^= 1
   return p
         
   
N = Ns * (N_data + Nrep * N_pre)
rnd.seed(36885555, version=2)
Preamble = rnd.choices(Symbols, weights=None, k=N_pre)
print(Preamble)
sd = tm.time_ns()
print("Rnd seed = {0}".format(sd))
rnd.seed(sd, version=2)
# Generate 1024 random bits
BitStream = rnd.choices([0, 1], weights=None, k= N_data)
for n in range(N_data-Npad, N_data, 1):
   BitStream[n] = 0
   
print(BitStream)
print(np.sum(BitStream))

# Encode bits
DataSymbol = np.zeros(N_data, dtype=np.intc)
DataTx = np.zeros((N, 3), dtype=np.float64)
for m in range(N_data):
   DataSymbol[m] = Encoder(BitStream[m])
   print("{0:>x}, {1:>x} Encoder state = {2:>x}, P1 = {3:>x}, P2 = {4:>x}".format(BitStream[m], DataSymbol[m], state, Parity(weight1 & state), Parity(weight2 & state)))

#Generate modulated QPSK signal   
p = 0

TxSignal = np.zeros(N, dtype=np.complex128)
t = np.zeros(N, dtype=np.float64)
S = np.zeros(N, dtype=np.complex128)
for n in range (N):
   if(n % Ns == 0):
      if(n < Nrep * (N_pre * Ns)):
         symb = Preamble[int(n / Ns) % N_pre]
      else:
         symb = Symbols[DataSymbol[p]]
         DataTx[p] = symb
         p += 1

   TxSignal[n] = np.complex(symb[0], symb[1])
# sigma^2 = N0*Bw/2Es, R approx 2*Bw
   BB_I = rnd.gauss(mu=symb[0], sigma=np.sqrt(Bw/(2.0*R*EsN0))) 
   BB_Q = rnd.gauss(mu=symb[1], sigma=np.sqrt(Bw/(2.0*R*EsN0))) 
#   print(n, BB_I, BB_Q)
   t[n] = n / (Ns * R)
   S[n] = np.complex(BB_I * np.cos(2.0 * np.pi * f_off * n / (Ns * R) + phi0) - \
       BB_Q * np.sin(2.0 * np.pi * f_off * n / (Ns * R) + phi0), \
       BB_Q * np.cos(2.0 * np.pi * f_off * n / (Ns * R) + phi0) + \
       BB_I * np.sin(2.0 * np.pi * f_off * n / (Ns * R) + phi0))
       
# Try to recover carrier offset frequency

I = np.complex(0.0, 1.0)
P1 = np.zeros(Ns*N_pre, dtype=np.complex128)
C = np.zeros(N, dtype=np.complex128)
Clk = np.zeros(N, dtype=np.float64)
Rx = np.zeros(N, dtype=np.complex128)
corr = 0.0
F0 = np.zeros(N, dtype=np.float64)
Buffer = np.zeros(Ns * N_pre, dtype = np.complex128)
Filter = np.zeros(N, dtype=np.complex128)
BBFil = 0.0+0.0j
DataRx = np.zeros((N, 3), dtype=np.float64)
P = 0.0
d = int(0)


for m in range(Ns*N_pre-1):
   P1[m] = np.complex(Preamble[int(m/Ns)][0], Preamble[int(m/Ns)][1])

F0[0] = 0.0
phi = 0.0   
dt = 1.0 / (Ns * R)
for n in range(N-1):
   LO = np.exp(-I*phi)
   phi += 2.0 * np.pi * dt * F0[n]
   if(int(n+Ns/2+d-9) % Ns < int(Ns/2)):
      Clk[n+1] = 1.0
   else:  
      Clk[n+1] = 0.0
       
# de-rotate
   Rx[n]= LO * S[n] * np.sqrt(R * EsN0 / Bw)
   BBFil += Rx[n]
# Generate carrier correction
   Ir = np.real(BBFil)
   Qr = np.imag(BBFil)
   err = Qr * np.sign(Ir) - Ir * np.sign(Qr)
   corr += err
# carrier AFC filter and loop
   Pr = Kp*err
   In = Ki * corr
   Fz = Pr + In
   F0[n+1] = Fz * G * R * Ns + F0[0]
#   print("Err = {0:>f}, corr = {1:>f}, Fz = {2:>f}, F0 = {3:>f}, phi = {4:>f}".format(err, corr, Fz, F0[n], phi))
   if(Clk[n+1] - Clk[n] < -0.1):
      BBFil = 0+0j
# Sliding correlator for preamble detection and clock phase correction  
   Buffer[n % (Ns*N_pre)] = Rx[n]
   C[n] = np.vdot(np.roll(P1, -n%(Ns*N_pre)), Buffer)/np.sqrt(np.vdot(P1, P1) * np.vdot(Buffer, Buffer))
# Clock recovery
   if(n > 2):
      if((np.abs(C[n-2] - C[n]) < np.abs(C[n-1]) * 0.05) and (np.abs(C[n-1]) > 0.25)):
         d = (n+int(Ns/2)) % Ns # Recover clock phase
         Ambig += C[n]/Nrep # Recovered carrier phase ambiguity
         print("Ambig = {0}, d = {1}".format(Ambig, d))
         if(np.abs(Ambig) < Eps):
            print("*** No preamble timing found! Try again! ***")
            sys.exit(-2)
# Integrate-and-dump filter
   if(n > Ns*N_pre*Nrep):
      Ifil = Ifil + Rx[n]
      Filter[n] = Ifil
#   print("n = {2}, Rx = {0}, Ifil = {1}".format(Rx[n], Ifil, n))
# Data symbol decision
   if((Clk[n+1] - Clk[n] < -0.1) and (n > Ns * Nrep * N_pre)):
      if np.abs(Ambig) < Eps:
         print("*** Preamble timing not found! Try again. F0={0:>f}".format(F0[n]))
         sys.exit(-1)
# Correct for carrier phase ambiguity and make decision
      I_comp = np.sign(np.real(Ifil * np.conjugate(Ambig/np.abs(Ambig))))
      Q_comp = np.sign(np.imag(Ifil * np.conjugate(Ambig/np.abs(Ambig))))     

      if(I_comp > 0.0):
         if(Q_comp > 0.0):
            DataRx[int((n-Ns*N_pre*Nrep)/Ns)] = Symbols[0]
         else:
            DataRx[int((n-Ns*N_pre*Nrep)/Ns)] = Symbols[1]
      else:
         if(Q_comp > 0.0):
            DataRx[int((n-Ns*N_pre*Nrep)/Ns)] = Symbols[3]
         else:
            DataRx[int((n-Ns*N_pre*Nrep)/Ns)] = Symbols[2]      

 #     print("Symbol {0} = Tx {1}, Rx {2}, I/Q decision = ({3} {4}) IaDfil = {5}, Clk[n+1] = {6}, Clk[n] = {7}".format(n/Ns, DataTx[int((n-Ns*Nrep*N_pre)/Ns-1)][2], DataRx[int((n-Ns*Nrep*N_pre)/Ns)][2], I_comp, Q_comp, Ifil*np.conjugate(Ambig)/np.abs(Ambig), Clk[n+1], Clk[n]))
      Ifil = 0.0+0.0j # Dump
   
 #  print(t[n], C[n], d)

RotA = np.conjugate(Ambig) / np.abs(Ambig)
sigma=np.sqrt(0.5 * Bw/(R*EsN0))
Path = Decoder(Filter*RotA/Ns, N, Ns, sigma)
NrMLErrs = 0
NrFECErrs = 0
for i in range(N_data-Npad):
   print("Bit no. = {0}, Tx bit = {1}, Rx bit w/FEC = {2}, Recovered hidden state = {3:>x}, dataTx= {4}, DataRx={5}".format(i, BitStream[i], Path[i+Nrep*N_pre+1]&0x1, Path[i+Nrep*N_pre+1], DataTx[i][2], DataRx[i+1][2]))
   if(BitStream[i] != Path[i+Nrep*N_pre+1]&0x1):
      NrFECErrs += 1
   if (DataTx[i][2] != DataRx[i+1][2]):
      NrMLErrs += 1
print("Number of symbol errors = {0}, Number of bit errors = {1}".format(NrMLErrs, NrFECErrs))
   

print("RotA = {0}".format(RotA))
fig, ax = plt.subplots(3, 1, facecolor='sienna')
ps1 = ax[0].plot(t[0:int(Ns*N/2)], np.real(Ns*Rx[0:int(Ns*N/2)]*RotA), 'r-', label='Rx real')
ps2 = ax[0].plot(t[0:int(Ns*N/2)], np.imag(Ns*Rx[0:int(Ns*N/2)]*RotA), 'b-', label='Rx imag')
ps3 = ax[0].plot(t[0:int(Ns*N/2)], np.abs(C[0:int(Ns*N/2)]), 'g-', label='Rake abs')
ps4 = ax[0].plot(t[0:int(Ns*N/2)], Ns*Clk[0:int(Ns*N/2)], 'y-', label='Clock')
ps5 = ax[0].plot(t[0:int(Ns*N/2)], np.real(Filter[0:int(Ns*N/2)]*RotA), lw = 2, c='brown', label='Real Fil')
ps6 = ax[0].plot(t[0:int(Ns*N/2)], np.imag(Filter[0:int(Ns*N/2)]*RotA), 'k-', label='Imag Fil')
pt1 = mp.Patch(color='red', label='Rx real')
pt2 = mp.Patch(color='blue', label='Rx imag')
pt3 = mp.Patch(color='green', label='Rake abs')
pt4 = mp.Patch(color='yellow', label='Clock')
pt5 = mp.Patch(color='brown', label='Real Fil')
pt6 = mp.Patch(color='black', label='Imag Fil')
l1 = ax[0].legend(handles=[pt1, pt2, pt3, pt4, pt5, pt6], loc='lower right', shadow=True)
ax[0].grid(True) 

#fig, ax = plt.subplots()
ps1 = ax[1].plot(t, np.real(Filter*RotA), 'r-', label='Rx real')
ps2 = ax[1].plot(t, np.imag(Filter*RotA), 'b-', label='Rx imag')
ps3 = ax[1].plot(t, np.real(0.5 * Ns *TxSignal), 'g-', label='Tx real')
ps4 = ax[1].plot(t, np.imag(0.5 * Ns*TxSignal), 'y-', label='Tx imag')
pt1 = mp.Patch(color='red', label='Rx real')
pt2 = mp.Patch(color='blue', label='Rx imag')
pt3 = mp.Patch(color='green', label='Tx Real')
pt4 = mp.Patch(color='yellow', label='Tx Imag')
l1 = ax[1].legend(handles=[pt1, pt2, pt3, pt4], loc='lower right', shadow=True)
ax[1].grid(True)

#ig, ax = plt.subplots()
ps4 = ax[2].plot(t[0:int(Ns*N/2)], Clk[0:int(Ns*N/2)], 'y-', label='Clock')
ps1 = ax[2].plot(t[0:int(Ns*N/2)], F0[0:int(Ns*N/2)], 'r-', label='Freq')
ps2 = ax[2].plot(t[0:int(Ns*N/2)], np.abs(C[0:int(Ns*N/2)]), 'b-', label='Corr')
pt1 = mp.Patch(color='red', label='Freq')
pt2 = mp.Patch(color='blue', label='Corr')
pt4 = mp.Patch(color='yellow', label='Clock Freq')
l1 = ax[2].legend(handles=[pt1, pt2,pt4], loc='lower right', shadow=True)
ax[2].grid(True)
plt.show()
   

sys.exit(1)

