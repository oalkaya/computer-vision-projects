import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please
    # Reshape A and B to have a new axis for broadcasting
    desc1_reshaped = desc1[:, np.newaxis, :]
    desc2_reshaped = desc2[np.newaxis, :, :]

    # Calculate the squared distance
    distances = np.sum((desc1_reshaped - desc2_reshaped)**2, axis=-1)
    
    return distances

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here
        # You may refer to np.argmin to find the index of the minimum over any axis

        # Find matching indices via argmin
        matches_1_to_2 = np.argmin(distances, axis=1)
        matches = np.vstack((np.arange(len(matches_1_to_2)), matches_1_to_2)).T

    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        # You may refer to np.min to find the minimum over any axis

        # Find the indices of the nearest neighbors for each keypoint in image 1 and 2
        matches_1_to_2 = np.argmin(distances, axis=1)
        matches_2_to_1 = np.argmin(distances, axis=0)
        # Identify mutual matches (Back and forth through the mapping ending with the same index order implies mutual match)
        mutual_matches_indices = np.where(np.arange(len(matches_1_to_2)) == matches_2_to_1[matches_1_to_2])[0]
        matches = np.vstack((mutual_matches_indices, matches_1_to_2[mutual_matches_indices])).T

    elif method == "ratio":
        # TODO: implement the ratio test matching here
        # You may use np.partition(distances,2,axis=0)[:,1] to find the second smallest value over a row

        # Find the indices of the nearest neighbors for each keypoint in image 1 (on image 2)
        nearest_neighbors = np.argmin(distances, axis=1)

        # Find the distances to the nearest neighbors
        nearest_distances = distances[np.arange(len(nearest_neighbors)), nearest_neighbors]

        # Find the second smallest distance for each keypoint
        second_smallest_distances = np.partition(distances, 2, axis=1)[:, 1]

        # Calculate the ratio between the distances of the 1st and 2nd nearest neighbors
        ratio = nearest_distances / second_smallest_distances

        # Filter matches based on the ratio threshold
        valid_matches = np.where(ratio < ratio_thresh)[0]

        # Create the array of matches
        matches = np.vstack((np.arange(len(nearest_neighbors))[valid_matches], nearest_neighbors[valid_matches])).T
        return matches

    else:
        raise NotImplementedError
    return matches

