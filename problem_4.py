"""
In this problem we are expanding on the DOA estimation from problem 3 to work with arbitrary 1D array.
The following code is only for azimuth estimation, not elevation.

We're doing a grid search over many possible angles.

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

# calculate a DOA spectrum d ∈ R N where di = |ai†s|, and then finds the peak in the spectrum

# steering matrix should have p rows and 480 columns
A = np.zeros((p, N), dtype=np.complex_)  # p rows for each antenna, N columns for each candidate angle

# initialize DOA spectrum d
d = np.zeros(N, dtype=np.float_)  # N elements for each candidate angle
# d will hold the magnitudes of the projections of s onto the steering vectors


for i in range(N): # build steering vector for every candidate angle theta i
    # calculate the steering vector for angle theta[i]
    theta_i_radians = np.deg2rad(theta[i])  # convert angle to radians
    A[:, i] = np.exp(-1j * 2 * np.pi * r / wavelength * np.sin(theta_i_radians))

    # calculate conjugate transpose of the steering vector
    A_conj_transpose = np.conj(A[:, i].T)

    # compute the dot product with the received signal s and store in spectrum d
    d[i] = np.abs(A_conj_transpose @ s2)  # dot product and take magnitude

# find the index of the maximum value in the DOA spectrum
max_index = np.argmax(d)
# find the corresponding angle
estimated_angle = theta[max_index]
# print the estimated angle
print(f"Estimated DOA angle: {estimated_angle:.2f} degrees")

# find estimated DOA angles for 4 targets using top 4 peaks in the DOA spectrum
peaks, _ = find_peaks(d, height=np.max(d) * 0.1) 
# sort peaks by their height
sorted_peaks = sorted(peaks, key=lambda x: d[x], reverse=True)[:4]  # take top 4 peaks
# extract estimated angles for the top 4 peaks
estimated_angles = theta[sorted_peaks]
print("Estimated DOA angles for the top 4 targets:")
for angle in estimated_angles:
    print(f"{angle:.2f} degrees")



"""
The signal vector s for a scene with multiple targets is provided in s_multi_target.txt.
Apply the extended algorithm you developed in part (b) to find and report the
number of targets K and their corresponding DOA estimates ˆθ1, · · · ,ˆθk
"""

# Plot the DOA spectrum d as a function of θi in db scale
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(theta, 20 * np.log10(d), label='DOA Spectrum')
plt.title('DOA Spectrum')
plt.xlabel('Angle (°)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.show()








# steering vector: why do we need it if we already have a vector of the phases from the recieved angle
# which corresponds to the angle of the target?






# 	After you do range-Doppler processing, you look at the bin where your target showed up (say: 15 m away, 3 m/s).
# You collect those P numbers into a column vector called s
# s is the actual phases # of the received signal at each antenna for that target, at that range and velocity.

# because s by itself is just a list of complex numbers; 
# to turn those into an angle you need to know exactly how those phases relate to geometry, noise, and ambiguity. The steering‐vector approach wraps all of that into one clean comparison. Here’s a more detailed breakdown:


# 	1.	Non-uniform spacing
# If your antennas aren’t equally spaced, each pair (p,q) has a different spacing r_q - r_p. 
# You’d need to solve a system of equations across many pairs and somehow fuse them.
