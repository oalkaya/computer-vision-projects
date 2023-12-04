import numpy as np

def estimate(particles, weights):
    """
    Estimate the state of the system as the weighted average of the particles.

    Parameters:
    - particles: The current set of particles, each representing a state estimate.
    - weights: The weights of each particle.

    Returns:
    - estimated_state: The estimated state as the weighted average of the particles.
    """
    estimated_state = np.average(particles, axis=0, weights=weights)
    return estimated_state