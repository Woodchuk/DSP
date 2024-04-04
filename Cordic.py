import numpy as np
import sys

# Generate tangent look-up table and normalizing constant
# Operating word length is 32 bits (-2147483248 < x < 2147483247)
# We use fixed point integer arithmetic suitable for hardware implementations
phi = np.zeros(32, dtype=np.int64)
K = 1.0
for i in range(32):
   a = 2.0 ** (-i)
   phi[i] = int(2147483648.0 * np.arctan(a)/np.pi)
   K = K / np.sqrt(1.0 + 2.0 ** (-2.0 * i))
Ki = int(2147483648 * K)
print(K, Ki)
for i in range(32):
   print("{0}  {1}".format(phi[i], 2.0**(-i)))
   
   
def Cordic(theta):
   n = 0
   s = np.zeros(21, dtype=np.int64)
   c = np.zeros(21, dtype=np.int64)
   t = np.zeros(21, dtype=np.int64)
   if (theta > -2147483648) and (theta < 2147483647):
      if (theta > 0): # Fix quadrant so only need to iterate over 90 degree range
         if theta < 1073741824:  # Quadrant 1
            s0 = 0
            c0 = Ki
            t[0] = 0
         else:  # Quadrant 2
            s0 = Ki
            c0 = 0
            t[0] = 1073741824
      else:
         if theta > -1073741824:
            s0 = -Ki
            c0 = 0
            t[0] = -1073741824
         else:
            s0 = 0
            c0 = -Ki
            t[0] = -2147483648
   else:
      print("theta is out of range!")
      sys.exit(-11)


   d = 1
   s[0] = s0
   c[0] = c0
   for n in range(20): #20 iterations to give around 1e-6 error
      if d > 0: # rotations for theta > present angle t[n]
         s[n+1] = s[n] + (c[n] >> n)
         c[n+1] = c[n] - (s[n] >> n)
         t[n+1] = t[n] + phi[n]
         d = np.sign(theta - t[n+1])
      else: # rotate in the negative sense if theta < t[n]
         s[n+1] = s[n] - (c[n] >> n)
         c[n+1] = c[n] + (s[n] >> n)
         t[n+1] = t[n] - phi[n]
         d = np.sign(theta - t[n+1])    
   return s[n+1], c[n+1]
   
   
for n in range(64):
   th =  -2147483647 + n * (2147483648 >> 5)
   s, c = Cordic(th)
   print("theta = {0}, s = {1}, c = {2}, sin = {3}, cos = {4}".format(180.0*th/2147483648.0, s/2147483648.0, c/2147483648.0, np.sin(np.pi*th/2147483648.0), np.cos(np.pi*th/2147483648.0)))
         
sys.exit(0)

