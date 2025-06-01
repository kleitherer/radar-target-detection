"""
Part C: DOA Algorithm for any antenna pattern
Before in part b, we processed the DOA of every detected target cell by taking the FFT of the range-Doppler across the antenna dimension.
However, this only worked for uniform linear array (ULA) where elements are equally spaced. We are expanding on this DOA estimation 
to work with any arbitrary 1D array (also, note that because this is 1D, the following algorithm is only for azimuth estimation, not elevation.)

Our algorithm:
In this problem, we're taking advantage of the fact that the phase of the received signal contains the DOA of the target.
Now, we're doing a grid search over many possible angles, ranging from -fov/2 to fov/2 degrees, and for each angle, we will compute a steering vector.
We then compare these steering vectors against the actual received phases to estimate the target’s direction.

In the problem, it defined the number of angles as N = 480, and the field of view as fov = 120 degrees.
Step 1: 
Build the steering matrix A where each column correponds to a steering vector for a candidate angle θi.
# The steering matrix A should have p rows (for each antenna) and N columns (for each candidate angle)
# where p is the number of antennas and N is the number of candidate angles.
# The steering vector for each candidate angle θi is given by:
# a_i = exp(-j * 2 * pi * r / λ * sin(θi))
# where r is the position of the antennas, λ is the wavelength, and θi is the candidate angle in radians.
you have to go through each candidate angle θi, calculate the steering vector a_i, and store it in the matrix A.




# steering vector: why do we need it if we already have a vector of the phases from the recieved angle
# which corresponds to the angle of the target?






# 	After you do range-Doppler processing, you look at the bin where your target showed up (say: 15 m away, 3 m/s).
# You collect those P numbers into a column vector called s
# s is the actual phases # of the received signal at each antenna for that target, at that range and velocity.

# because s by itself is just a list of complex numbers; 
# to turn those into an angle you need to know exactly how those phases relate to geometry, noise, and ambiguity. The steering‐vector approach wraps all of that into one clean comparison. Here’s a more detailed breakdown:




# Step 2: Calculate the DOA spectrum d ∈ R^N where di = |ai†s|, and then find the peak in the spectrum.


Part D: Extend DOA Algorithm to Multiple Targets

Apply the extended algorithm you developed in part (b) to find and report the
number of targets K and their corresponding DOA estimates ˆθ1, · · · ,ˆθk

"""
import numpy as np
from scipy.signal import find_peaks

s = np.loadtxt('data/problem_4/s_single_target.txt', dtype=np.complex_) # measured phase and amplitude of the received signal at each antenna for a target at a specific range and velocity
s2 = np.loadtxt('data/problem_4/s_multi_target.txt', dtype=np.complex_)
r = np.loadtxt('data/problem_4/array_elem_pos_mm.txt')
p = r.shape[0] # number of antennas
N = 480
fov = 120 # field of view (degrees)
theta = np.linspace(-fov/2, fov/2, N)  # array of candidate angles (from -60 to 60 degrees)
wavelength = 3.89 # wavelength (mm)

"Step 1: Build the steering matrix A where each column corresponds to a steering vector for a candidate angle θi."
A = np.zeros((p, N), dtype=np.complex_)
d = np.zeros(N, dtype=np.float_) 


for i in range(N): # build steering vector for every candidate angle theta i
    theta_i_radians = np.deg2rad(theta[i]) 
    A[:, i] = np.exp(-1j * 2 * np.pi * r / wavelength * np.sin(theta_i_radians))
    A_conj_transpose = np.conj(A[:, i].T)
    d[i] = np.abs(A_conj_transpose @ s2) 

max_index = np.argmax(d)
estimated_angle = theta[max_index]
print(f"Estimated DOA angle: {estimated_angle:.2f} degrees")

# find estimated DOA angles for 4 targets using top 4 peaks in the DOA spectrum
peaks, _ = find_peaks(d, height=np.max(d) * 0.1) 
sorted_peaks = sorted(peaks, key=lambda x: d[x], reverse=True)[:4] 
estimated_angles = theta[sorted_peaks]
print("Estimated DOA angles for the top 4 targets:")
for angle in estimated_angles:
    print(f"{angle:.2f} degrees")


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(theta, 20 * np.log10(d), label='DOA Spectrum')
plt.title('DOA Spectrum')
plt.xlabel('Angle (°)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.show()
