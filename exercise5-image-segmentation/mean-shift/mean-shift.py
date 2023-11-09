import time
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale


# Changed function signature to compute vectorized distances
def distance(X):
    return np.sqrt(((X[:, np.newaxis] - X) ** 2).sum(axis=2)) # Vectorized computation of all pairwise distances

def gaussian(dists, bandwidth):
    return np.exp(-0.5 * (dists / bandwidth) ** 2)

def update_point(weights, X):
    weighted_sum = np.dot(weights, X)
    sum_of_weights = weights.sum(axis=1, keepdims=True)
    return weighted_sum / sum_of_weights

def meanshift_step(X, bandwidth=2.5):
    dists = distance(X)                     # Use the vectorized distance function
    weights = gaussian(dists, bandwidth)    # Use the gaussian function
    new_X = update_point(weights, X)        # Use the vectorized update function
    return new_X                            # Return new X


def meanshift(X, bandwidth, num_labels):
    # Set number of centroids to num of pixels at the start (set it to some large value)
    num_centroids = X.shape[0]
    step_count = 0
    while True:
        X = meanshift_step(X, bandwidth)
        # Check for convergence behaviour via centroids
        centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)
        num_centroids = centroids.shape[0]
        print("Step " + str(step_count+1) + " (centroids=" + str(num_centroids) + ")")
        if(num_centroids <= num_labels):
            break
        else:
            step_count += 1
    return X

# Hyperparameters
scale = 0.5         # downscale the image to run faster
bandwidth = 3       # bandwitdth for gaussians

# Load image and convert it to CIELAB space
image = rescale(io.imread('exercise5-image-segmentation/mean-shift/eth.jpg'), scale, channel_axis=-1)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Load label colours
colors = np.load('exercise5-image-segmentation/mean-shift/colors.npz')['colors']

# Run your mean-shift algorithm
t = time.time()
X = meanshift(image_lab, bandwidth, colors.shape[0]) # Use number of label colours as loop condition within mean shift
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))



# Draw labels as an image
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, channel_axis=-1)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('exercise5-image-segmentation/mean-shift/result.jpg', result_image)
