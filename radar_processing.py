import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# load the dataset
dataset = np.load('data/problem_3/radar_imaging_raw_data.npz')
radar_raw_data_cube = np.squeeze(dataset['raw_tdm_frame']) # raw data cube
f_s = dataset['adc_sampling_freq']                         # ADC sampling rate (Hz)
num_samp_per_chirp = dataset['tdm_num_samples_per_chirp']  # number of samples per chirp
num_chirp_per_frame = dataset['tdm_num_blocks']            # number of chirps per frame
num_rx_antenna = dataset['n_rx']                           # number of RX antennas
num_tx_antenna = dataset['n_tx']                           # number of TX antennas
chirp_duration = dataset['tdm_chirp_duration']             # chirp duration (s)
slew_rate = 9.994e6/1e-6                                   # chirp slew rate (Hz/s)
c_air = 299702547.236                                      # speed of light in air (m/s)
wavelength = c_air/dataset['carrier_freq']                 # carrier wavelength (m)
antenna_spacing = wavelength/2                             # distance of adjacent antennas

print(f"---Data cube dimensions---")
print(f"- RX antennas: {radar_raw_data_cube.shape[0]}")
print(f"- Chirps per frame (doppler bins): {radar_raw_data_cube.shape[1]}")  
print(f"- Samples per chirp (range bins): {radar_raw_data_cube.shape[2]}")


"""
Part a: Process the raw radar data to create a range-Doppler map and visualize it.

The raw data cube is stored in the 3D array radar_raw_data_cube with dimensions
    [num_rx_antenna, num_chirp_per_frame, num_samp_per_chirp]

Step 1: Calculate range and velocity bins based on radar parameters.
    Our final RD map should have dimensions: MxN, where:
        Doppler bins (M) = each chirp is what we want to convert to frequency domain
        Range bins (N) = chirp duration * sampling rate = samples per chirp, which is what we convert to frequency domain

Step 2: Process the raw data cube to create a range-Doppler map for each RX antenna.
    - Loop through each RX antenna and perform either 1D FFT on the data cube.
        - The first FFT will be across the sample index (n) to extract range.
        - The second FFT will be across the chirp index (m) to extract Doppler.
    - Alternatively, you can also perform a 2D FFT on the data cube.
    - Store the resulting range-Doppler maps in an array. The array should have 
    the same dimensions as the radar_raw_data_cube and allowing complex numbers

Step 3: Average the range-Doppler maps across RX antennas.
    - Calculate the magnitude squared of all range-Doppler maps.
    - Average across RX antennas to go from 3D to 2D.
    - Visualize the averaged range-Doppler map using a heatmap function plot_RD_heatmap.

"""

"Step 1: Calculate range and velocity bins based on radar parameters."

range_bins_main = np.arange(int(num_samp_per_chirp)) * f_s * c_air / (2 * num_samp_per_chirp * slew_rate)
doppler_bins = np.fft.fftfreq(int(num_chirp_per_frame), chirp_duration)
doppler_bins_shifted = np.fft.fftshift(doppler_bins) 
velocity_bins_main = doppler_bins_shifted * wavelength / 2 # translate to physical values

def plot_RD_heatmap(averaged_RD_map, range_bins_main, velocity_bins_main):
    plt.figure(figsize=(12, 8))
    plt.imshow(averaged_RD_map, 
              aspect='auto',
              extent=[range_bins_main[0], range_bins_main[-1], 
                     velocity_bins_main[0], velocity_bins_main[-1]],
              origin='lower',
              cmap='viridis')
    
    plt.xlabel('Range (m)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Range-Doppler Map')
    
    plt.show()

"Step 2: Process the raw data cube to create a range-Doppler map for each RX antenna."

all_RD_maps = np.zeros(radar_raw_data_cube.shape, dtype=complex)

for i in range(num_rx_antenna): # iterate through each RX antenna
    # Version 1: take 1D FFT across sample index n to extract range 
    # then take 1D FFT across chirp index m to extract doppler
    single_Rd = np.fft.fft(radar_raw_data_cube[i,:,:], axis=1)
    single_RD_unshifted = np.fft.fft(single_Rd, axis=0)
    single_RD = np.fft.fftshift(single_RD_unshifted, axes=0) #center the zero frequency component
    
    # Version 2: taking 2D FFT 
        # single_RD_unshifted = np.fft.fft2(radar_raw_data_cube[i,:,:])
        # single_RD = np.fft.fftshift(single_RD_unshifted)

    all_RD_maps[i] = single_RD


"Step 3: Average the range-Doppler maps across RX antennas."

magnitude_squared = np.sqrt(np.abs(all_RD_maps)**2) # calculate the magnitude squared of all range-Doppler maps

averaged_RD_map = np.mean(magnitude_squared, axis=0) # average across rx antennas to go from 3D to 2D

# if you want to see RD map on a dB scale:
# averaged_RD_map = 10 * np.log10(averaged_RD_map) 
plot_RD_heatmap(averaged_RD_map, range_bins_main, velocity_bins_main)

"""
Part b: Target detection and DOA estimation for 1D ULA geometry.

Our next goal is to not only be able to visualize the RD map, but to understand what is physically in our scene. 
So, we'll implement a target detection algorithm to identify targets in the RD map and estimate their direction of arrival (DOA) angles.
Then, we'll visualize the targets in a 2D pointcloud and compare to the ground truth.

Step 1: Detect targets in the averaged range-Doppler map using a threshold.
    - Use a threshold to identify potential targets in the RD map. We were given a threshold of 99.6 percentile.
    - Using argwhere, find the indices of the detected targets in the RD map. 

Step 2: For each target index, find its DOA elevation angle using the range-Doppler map and find peaks.
   - For each detected target, we use the indices to plug into the RD map and extract physical range and velocity values of interest.
   - We only want to find the DOA elevation angle at that range and velocity. So we extract the RD map at that range and velocity, and
   perform a 1D FFT to find the DOA elevation angle.
   - Then use find_peaks to find the peaks in the DOA spectrum, which correspond to the DOA angles.

Step 3:  Convert polar coordinates to cartesian for 2D plot of target locations.
    - Iterate through each point in the point cloud list, which contains range, velocity, and angle.
    - Convert polar coordinates (range, angle) to cartesian coordinates (x, y).
"""

"Step 1: Detect targets in the averaged range-Doppler map using a threshold."

threshold = np.percentile(averaged_RD_map, 99.6)
detected_indices = np.argwhere(averaged_RD_map > threshold) # detected_indices is a 2D array where each row is [doppler_index, range_index]

"Step 2:  For each target index, find its DOA elevation angle using the range-Doppler map and find peaks."

pcloud_list = []

for target in detected_indices: 
    doppler_index = target[0]  
    range_index = target[1]
    
    range = range_bins_main[range_index]
    velocity = velocity_bins_main[doppler_index]
    
    RDa = all_RD_maps[:, doppler_index, range_index]
    
    doa_fft = np.fft.fft(RDa)
    doa_fft_shifted = np.fft.fftshift(doa_fft)
    doa_spectrum = np.abs(doa_fft_shifted)
    doa_peaks, _ = find_peaks(doa_spectrum, height=np.percentile(doa_spectrum, 97.5))
    
    angle_bins = np.arcsin(np.linspace(-1, 1, len(doa_spectrum))) * 180/np.pi
    
    for peak_idx in doa_peaks:
        angle = angle_bins[peak_idx]
        pcloud_list.append((range, velocity, angle))


"Step 3: Convert polar coordinates to cartesian for 2D plot of target locations."

x_coords = []
y_coords = []
# velocities = []

for point in pcloud_list:
    range_val = point[0]    # first element is range
    # velocity = point[1]     # second element is velocity
    angle = point[2]        # third element is angle

    angle_rad = np.deg2rad(angle)
    
    x = range_val * np.sin(angle_rad)
    y = range_val * np.cos(angle_rad)
    
    x_coords.append(x)
    y_coords.append(y)
    # velocities.append(velocity)

x_coords = np.array(x_coords)
y_coords = np.array(y_coords)
# velocities = np.array(velocities)


plt.figure(figsize=(10, 10))
plt.scatter(x_coords, y_coords, cmap='coolwarm', s=100)
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title(f'Reconstructed 2D Point Cloud of Targets')
plt.grid(True)
plt.axis('equal')
plt.show()
