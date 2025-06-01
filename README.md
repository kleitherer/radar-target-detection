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

    
## Part C: DOA Algorithm for any antenna pattern.

Before in part b, we processed the DOA of every detected target cell by taking the FFT of the range-Doppler across the antenna dimension.
However, this only worked for uniform linear array (ULA) where elements are equally spaced. We are expanding on this DOA estimation 
to work with any arbitrary 1D array (also, note that because this is 1D, the following algorithm is only for azimuth estimation, not elevation.)

### The algorithm: 

The phase of the received signal contains information about the direction of arrival (DOA). To extract this, we perform a grid search over many possible angles. For each candidate angle, we compute a steering vector which models the expected phase shifts across antennas if the target was located at that angle. We then compare these steering vectors to the actual received phases to estimate the target’s direction by taking the inner product (i.e. the correlation) between the two phases.

    The problem specifies: 
      - Number of candidate angles: N = 480
      - Field of view: FOV = 120 degrees (angles range from -60° to +60°) 

Step 1: Build the steering matrix A where each column correponds to a steering vector for a candidate angle θi.

    The steering matrix A should have p rows (for each antenna) and N columns (for each candidate angle)
    where p is the number of antennas and N is the number of candidate angles.
    
    The steering vector for each candidate angle θi is given by:
    
  $$
  a_i = \exp\left(-j \cdot \frac{2\pi}{\lambda} \cdot r \cdot \sin(\theta_i)\right)
  $$

    where:
      r = array of antenna positions (length p)
      λ = wavelength
      θᵢ = candidate angle in radians
      
    Loop through each candidate angle θᵢ, calculate the steering vector aᵢ, and store it as column i in matrix A.

Step 2: Compute DOA spectrum using a matched filter.

The DOA spectrum helps us answer how well a specific steering vector matches our measured data. To break this down using math, we begin with the dot product since it measures how aligned two vectors are. For the complex space, we need to use the conjugate transpose to account for phase shifts. Because we care about how strongly aligned they are, we need to take the magnitude (i.e. absolute value). The index with the largest magnitude indicates we have a certain angle that we can say is coming from the target. So the equation for each index in the DOA spectrum is:

$$
d_i = |(a_iᴴ * s) |
$$

    where:
      a_iᴴ is the conjugate transpose of the steering vector at one of the 480 angles we estimate
      s is the vector of measured phases.

Step 3: Find the peak in the DOA spectrum.

    Find the index within the DOA spectrum that has the highest magnitude.
    This index is the angle at which the target is coming from.

Step 4: Expand this algorithm for multiple targets.

    Instead of finding one global maximum across the DOA spectrum, expand the search to include next highest correlation values.
