import numpy as np

def filter_keypoints(img, keypoints, patch_size = 9):
    '''
    Inputs:
    - keypoints:    (q, 2) numpy array of keypoint locations [x, y]
    Returns:
    - keypoints:    (q', 2) numpy array of keypoint locations [x, y] that are far enough from edges
    '''
    # TODO: Filter out keypoints that are too close to the edges

    # Define the pixel threshold from patch size
    offset = int(np.floor(patch_size / 2.0))

    # Extract row and column coordinates
    rows, cols = keypoints[:, 0], keypoints[:, 1]

    # Check if keypoints are within the specified distance from the image edges
    valid_keypoints_mask = (rows > offset) & (rows < img.shape[1] - offset) & (cols > offset) & (cols < img.shape[0] - offset)
    # Apply mask
    filtered_keypoints = keypoints[valid_keypoints_mask]

    return filtered_keypoints

# The implementation of the patch extraction is already provided here
def extract_patches(img, keypoints, patch_size = 9):
    '''
    Extract local patches for each keypoint
    Inputs:
    - img:          (h, w) gray-scaled images
    - keypoints:    (q, 2) numpy array of keypoint locations [x, y]
    - patch_size:   size of each patch (with each keypoint as its center)
    Returns:
    - desc:         (q, patch_size * patch_size) numpy array. patch descriptors for each keypoint
    '''
    h, w = img.shape[0], img.shape[1]
    img = img.astype(float) / 255.0
    offset = int(np.floor(patch_size / 2.0))
    ranges = np.arange(-offset, offset + 1)
    desc = np.take(img, ranges[:,None] * w + ranges + (keypoints[:, 1] * w + keypoints[:, 0])[:, None, None]) # (q, patch_size, patch_size)
    desc = desc.reshape(keypoints.shape[0], -1) # (q, patch_size * patch_size)
    return desc

