import numpy as np

def resample(particles, weights):
    """
    Resample the particles based on their weights.

    Parameters:
    - particles: The current set of particles, each representing a state estimate.
    - weights: The normalized weights of each particle.

    Returns:
    - new_particles: The resampled set of particles.
    """
    # Number of particles
    num_particles = len(weights)
    
    # Cumulative sum of the weights
    cumulative_sum = np.cumsum(weights)
    
    # Uniformly spaced sequence of random numbers
    random_offsets = np.random.rand(num_particles)
    
    # Initialize the indices array
    indices = np.zeros(num_particles, 'i')
    
    # Resample the particles
    for i, rand in enumerate(random_offsets):
        indices[i] = np.searchsorted(cumulative_sum, rand)
        
    # Construct the new set of particles
    new_particles = particles[indices]
    
    return new_particles