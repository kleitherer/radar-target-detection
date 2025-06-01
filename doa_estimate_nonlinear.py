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
