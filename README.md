# radar-target-detection

The following is included in radar_processing.py 
## Part A: Process raw radar data into Range-Doppler map.

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

## Part B: Target detection and DOA estimation for 1D ULA geometry.

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
