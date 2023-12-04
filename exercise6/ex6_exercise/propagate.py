import cv2
import numpy as np

def propagate(particles, frame_height, frame_width, params):
    """
    Propagate the particles given the system prediction model (matrix A) and
    the system noise represented by params.sigma_position and params.sigma_velocity.

    Parameters:
    - particles: The current set of particles, each representing a state estimate.
    - frame_height: The height of the frame to ensure particles stay within bounds.
    - frame_width: The width of the frame to ensure particles stay within bounds.
    - params: Dictionary of parameters, including model type and noise parameters.

    Returns:
    - particles: The set of propagated particles.
    """
    A = None
    Q = None

    if params["model"] == 0:  # No motion model
        # State transition matrix A for no motion
        A = np.eye(2)  # 2x2 identity matrix for the position-only state vector
        
        # Process noise covariance matrix Q = B*B.T
        Q = np.diag([params["sigma_position"]**2, params["sigma_position"]**2])

    elif params["model"] == 1:  # Constant velocity model
        # State transition matrix A for constant velocity
        A = np.array([[1, 0, 1, 0],  # Assuming a unit time step
                      [0, 1, 0, 1],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        # Process noise covariance matrix Q = B*B.T
        Q = np.diag([params["sigma_position"]**2, params["sigma_position"]**2,
                     params["sigma_velocity"]**2, params["sigma_velocity"]**2])

    # Propagate each particle
    for i in range(particles.shape[0]): # Iterate for number of particles
        noise = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q) # Directly use Q to generate noise
        particles[i] = A @ particles[i] + noise # Shorthand for np.dot

        # Ensure particles are within the frame bounds
        particles[i, 0] = np.clip(particles[i, 0], 0, frame_width - 1)
        particles[i, 1] = np.clip(particles[i, 1], 0, frame_height - 1)

    return particles