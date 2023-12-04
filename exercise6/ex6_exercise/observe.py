import numpy as np
from color_histogram import color_histogram
from chi2_cost import chi2_cost

def observe(particles, frame, bbox_height, bbox_width, hist_bin, target_hist, sigma_observe):
    """
    Parameters:
    - particles: The current set of particles, each representing a state estimate.
    - frame: The current frame from the video.
    - bbox_height, bbox_width: The size of the bounding box.
    - hist_bin: The number of bins for the histogram.
    - target_hist: The target color histogram to compare against.
    - sigma_observe: The standard deviation of the observation model noise.

    Returns:
    - weights: The normalized weights for each particle.
    """
    num_particles = particles.shape[0]
    weights = np.zeros(num_particles)
    
    for i in range(num_particles):
        # Extract the particle's predicted position
        x, y = int(particles[i, 0]), int(particles[i, 1])
        
        # Compute the color histogram for the predicted position
        particle_hist = color_histogram(x, y, x+bbox_width, y+bbox_height, frame, hist_bin)
        
        # Calculate the chi-squared distance between particle histogram and target histogram
        distance = chi2_cost(particle_hist, target_hist)
        
        # Convert distance to weight, assuming a Gaussian error model
        weights[i] = np.exp(-0.5 * (distance**2) / (sigma_observe**2))
    
    # Normalize the weights
    weights /= np.sum(weights)
    
    return weights