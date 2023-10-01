import numpy as np
import cv2
from scipy import signal #for the scipy.signal.convolve2d function
from scipy import ndimage #for the scipy.ndimage.maximum_filter

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # 1. Compute image gradients in x and y direction
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.

    sobel_x = np.array([[-1, 0, 1], # Try cv2.Sobel directly?
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    # Convolve the image with the Sobel kernels
    Ix = signal.convolve2d(img, sobel_x, mode='same', boundary='symm')
    Iy = signal.convolve2d(img, sobel_y, mode='same', boundary='symm')
    
    # 2. Blur the computed gradients
    # TODO: compute the blurred image gradients
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)

    Ix_blurred  = cv2.GaussianBlur(src=Ix * Ix, ksize=(0,0), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    Iy_blurred  = cv2.GaussianBlur(src=Iy * Iy, ksize=(0,0), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    Ixy_blurred = cv2.GaussianBlur(src=Ix * Iy, ksize=(0,0), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)

    # 3. Compute elements of the local auto-correlation matrix "M"
    # TODO: compute the auto-correlation matrix here
    det_M = Ix_blurred * Iy_blurred - Ixy_blurred**2
    trace_M = Ix_blurred + Iy_blurred

    # 4. Compute Harris response function C
    # TODO: compute the Harris response function C here
    C = det_M - k * trace_M**2

    # 5. Detection with threshold and non-maximum suppression
    # TODO: detection and find the corners here
    # For the non-maximum suppression, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    # You may refer to np.where to find coordinates of points that fulfill some condition; Please, pay attention to the order of the coordinates.
    # You may refer to np.stack to stack the coordinates to the correct output format

    # Do thresholding to get rid of small values (set to 0)
    C_thresh = np.copy(C)
    C_thresh[C < thresh] = 0 

    # Apply maximum filter to find the maximum response in each 3x3 window
    max_responses = ndimage.maximum_filter(C_thresh, size=3, mode='constant')

    # Loop over the whole image to eliminate candidates.
    C_supp = np.zeros_like(C_thresh)
    for i in range(1, C_thresh.shape[0] - 1):
        for j in range(1, C_thresh.shape[1] - 1):
            # Check if the center pixel is smaller than another pixel in the window
            if(C_thresh[i, j] > thresh):
                if C_thresh[i, j] == max_responses[i, j]:
                    C_supp[i, j] = 1

    # Get corner list
    corners = np.vstack(np.where(C_supp == 1)[::-1]).T                
    return corners, C

