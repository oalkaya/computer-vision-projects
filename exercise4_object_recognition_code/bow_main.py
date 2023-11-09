import numpy as np
import cv2
import glob
import os
from sklearn.cluster import KMeans
from tqdm import tqdm

# from matplotlib import pyplot as plt


def findnn(D1, D2):
    """
    :param D1: NxD matrix containing N feature vectors of dim. D
    :param D2: MxD matrix containing M feature vectors of dim. D
    :return:
        Idx: N-dim. vector containing for each feature vector in D1 the index of the closest feature vector in D2.
        Dist: N-dim. vector containing for each feature vector in D1 the distance to the closest feature vector in D2
    """
    N = D1.shape[0]
    M = D2.shape[0]  # [k]

    # Find for each feature vector in D1 the nearest neighbor in D2
    Idx, Dist = [], []
    for i in range(N):
        minidx = 0
        mindist = np.linalg.norm(D1[i, :] - D2[0, :])
        for j in range(1, M):
            d = np.linalg.norm(D1[i, :] - D2[j, :])

            if d < mindist:
                mindist = d
                minidx = j
        Idx.append(minidx)
        Dist.append(mindist)
    return Idx, Dist


def grid_points(img, nPointsX, nPointsY, border):
    """
    :param img: input gray img, numpy array, [h, w]
    :param nPointsX: number of grids in x dimension
    :param nPointsY: number of grids in y dimension
    :param border: leave border pixels in each image dimension
    :return: vPoints: 2D grid point coordinates, numpy array, [nPointsX*nPointsY, 2]
    """
    h, w = img.shape

    # Generate a grid of points at the corners of each cell.
    x_coords, y_coords = np.meshgrid(
        np.linspace(border, w - border, nPointsX),
        np.linspace(border, h - border, nPointsY)
    )
    
    # Stack the x and y coordinates into a single array.
    vPoints = np.column_stack((x_coords.ravel(), y_coords.ravel()))

    return vPoints



def descriptors_hog(img, vPoints, cellWidth, cellHeight):
    nBins = 8
    w = cellWidth
    h = cellHeight

    grad_x = cv2.Sobel(img, cv2.CV_32F, dx=1, dy=0, ksize=1) # Change type to cv2.CV_32F for cartToPolar
    grad_y = cv2.Sobel(img, cv2.CV_32F, dx=0, dy=1, ksize=1) # Change type to cv2.CV_32F for cartToPolar

    descriptors = []  # list of descriptors for the current image, each entry is one 128-d vector for a grid point
    for i in range(len(vPoints)):
        center_x = round(vPoints[i, 0])
        center_y = round(vPoints[i, 1])

        desc = []
        for cell_y in range(-2, 2):
            for cell_x in range(-2, 2):
                start_y = center_y + (cell_y) * h
                end_y = center_y + (cell_y + 1) * h

                start_x = center_x + (cell_x) * w
                end_x = center_x + (cell_x + 1) * w

                # todo
                # compute the angles
                # compute the histogram
                
                # Crop the gradient magnitude and orientation within the cell.
                cell_grad_x = grad_x[start_y:end_y, start_x:end_x]
                cell_grad_y = grad_y[start_y:end_y, start_x:end_x]

                # Calculate gradient magnitude and orientation.
                mag, angle = cv2.cartToPolar(cell_grad_x, cell_grad_y)

                # Compute the histogram of gradient orientations within the cell.
                hist, _ = np.histogram(angle, bins=nBins, range=(0, 2 * np.pi), weights=mag)

                desc.extend(hist)

        descriptors.append(desc)

    descriptors = np.asarray(descriptors) # [nPointsX*nPointsY, 128], descriptor for the current image
    return descriptors




def create_codebook(nameDirPos, nameDirNeg, k, numiter):
    """
    :param nameDirPos: dir to positive training images
    :param nameDirNeg: dir to negative training images
    :param k: number of kmeans cluster centers
    :param numiter: maximum iteration numbers for kmeans clustering
    :return: vCenters: center of kmeans clusters, numpy array, [k, 128]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDirPos, '*.png')))
    vImgNames = vImgNames + sorted(glob.glob(os.path.join(nameDirNeg, '*.png')))

    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    vFeatures = []  # list for all features of all images (each feature: 128-d, 16 histograms containing 8 bins)
    # Extract features for all image
    for i in tqdm(range(nImgs)):
        # print('processing image {} ...'.format(i+1))
        img = cv2.imread(vImgNames[i])  # [172, 208, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]

        # Collect local feature points for each image, and compute a descriptor for each local feature point
        vPoints = grid_points(img, nPointsX, nPointsY, border)
        descriptors = descriptors_hog(img, vPoints, cellWidth, cellHeight)
        
        # Append to vFeatures to do K-Means over all images, instead of doing it individually
        vFeatures.extend(descriptors)


    vFeatures = np.asarray(vFeatures)  # [n_imgs, n_vPoints, 128]
    vFeatures = vFeatures.reshape(-1, vFeatures.shape[-1])  # [n_imgs*n_vPoints, 128]
    print('number of extracted features: ', len(vFeatures))


    # Cluster the features using K-Means
    print('clustering ...')
    kmeans_res = KMeans(n_clusters=k, max_iter=numiter, n_init=10).fit(vFeatures)
    vCenters = kmeans_res.cluster_centers_  # [k, 128]
    return vCenters


def bow_histogram(vFeatures, vCenters):
    """
    :param vFeatures: MxD matrix containing M feature vectors of dim. D
    :param vCenters: NxD matrix containing N cluster centers of dim. D
    :return: histo: N-dim. numpy vector containing the resulting BoW activation histogram.
    """
    histo = np.zeros(vCenters.shape[0])  # Initialize the histogram with zeros, one bin for each cluster center

    # Loop over each feature vector in vFeatures
    for feature in vFeatures:
        # Compute the euclidian distances between the feature vector and all cluster centers
        distances = [np.linalg.norm(feature - center) for center in vCenters]

        # Find the index of the nearest cluster center (the "visual word")
        nearest_center_index = np.argmin(distances)

        # Increment the corresponding bin in the histogram
        histo[nearest_center_index] += 1

    return histo





def create_bow_histograms(nameDir, vCenters):
    """
    :param nameDir: dir of input images
    :param vCenters: kmeans cluster centers, [k, 128] (k is the number of cluster centers)
    :return: vBoW: matrix, [n_imgs, k]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDir, '*.png')))
    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    # Extract features for all images in the given directory
    vBoW = []
    for i in tqdm(range(nImgs)):
        # print('processing image {} ...'.format(i + 1))
        img = cv2.imread(vImgNames[i])  # [172, 208, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]

        # Collect local feature points for each image, and compute a descriptor for each local feature point
        vPoints = grid_points(img, nPointsX, nPointsY, border)
        descriptors = descriptors_hog(img, vPoints, cellWidth, cellHeight)

        # Calculate the BoW histogram for this image using the previously defined function
        histo = bow_histogram(descriptors, vCenters)

        vBoW.append(histo)

    vBoW = np.asarray(vBoW)  # [n_imgs, k]
    return vBoW



def bow_recognition_nearest(histogram,vBoWPos,vBoWNeg):
    """
    :param histogram: bag-of-words histogram of a test image, [1, k]
    :param vBoWPos: bag-of-words histograms of positive training images, [n_imgs, k]
    :param vBoWNeg: bag-of-words histograms of negative training images, [n_imgs, k]
    :return: sLabel: predicted result of the test image, 0(without car)/1(with car)
    """

    # Calculate distances between the test histogram and histograms of positive and negative training examples
    DistPos = np.linalg.norm(histogram - vBoWPos, axis=1)  # Euclidean distances to positive training histograms
    DistNeg = np.linalg.norm(histogram - vBoWNeg, axis=1)  # Euclidean distances to negative training histograms

    # Find the nearest neighbor in the positive and negative sets
    min_dist_pos = np.min(DistPos)
    min_dist_neg = np.min(DistNeg)

    # Decide based on the nearest neighbor
    if min_dist_pos < min_dist_neg:
        sLabel = 1  # Car
    else:
        sLabel = 0  # No car

    return sLabel


# HELPER FUNCTION FOR SANITY CHECKING MY OWN RESULTS

# def save_images_with_overlays(nameDir_test, inference_dir, vBoWTest, vBoWPos, vBoWNeg):

#     nPointsX = 10
#     nPointsY = 10
#     border = 8

#     os.makedirs(inference_dir, exist_ok=True)

#     # Get image paths for test samples
#     vImgNamesTest = sorted(glob.glob(os.path.join(nameDir_test, '*.png')))

#     for i, image_path in enumerate(vImgNamesTest):
#         img = cv2.imread(image_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale [h, w]
#         img_with_overlay = img.copy()

#         # Get grid points using the grid_points function
#         vPoints = grid_points(img, nPointsX, nPointsY, border)

#         # Back to BGR
#         img_with_overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

#         # Overlay grid points
#         for point in vPoints:
#             x, y = point
#             cv2.circle(img_with_overlay, (int(x), int(y)), 1, (0, 0, 127), -1)

#         # Calculate the width of each histogram bar
#         bar_width = 1

#         # Determine the x-coordinates for each bar in the histogram visualization
#         x_coords = np.arange(len(vBoWTest[i]))

#         # Calculate distances between the test histogram and histograms of positive and negative training examples
#         DistPos = np.linalg.norm(vBoWTest[i] - vBoWPos, axis=1)
#         DistNeg = np.linalg.norm(vBoWTest[i] - vBoWNeg, axis=1)

#         # Find the nearest neighbor in the positive and negative sets
#         nearest_pos_index = np.argmin(DistPos)
#         nearest_neg_index = np.argmin(DistNeg)

#         # Find the nearest neighbor distances in the positive and negative sets
#         min_dist_pos = np.min(DistPos)
#         min_dist_neg = np.min(DistNeg)

#         # Determine if it's positive or negative based on which distance is smaller
#         if min_dist_pos < min_dist_neg:
#             label = 'Positive'
#             text_color = (0, 255, 0)
#         else:
#             label = 'Negative'
#             text_color = (0, 0, 255)

#         # Draw text on the image
#         cv2.putText(img_with_overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

#         # Create a new figure for the visualization
#         fig, ax = plt.subplots(1, 4, figsize=(20, 5))

#         # Display the original image with grid points
#         ax[0].imshow(cv2.cvtColor(img_with_overlay, cv2.COLOR_BGR2RGB))
#         ax[0].set_title('Original Image')

#         # Create a bar graph for the BoW histogram
#         ax[1].bar(x_coords, vBoWTest[i], bar_width, color='black')
#         ax[1].set_xlabel('Visual Word')
#         ax[1].set_ylabel('Frequency')
#         ax[1].set_title('BoW Histogram')

#         # Create a bar graph for the nearest positive training histogram
#         ax[2].bar(x_coords, vBoWPos[nearest_pos_index], bar_width, color='g')
#         ax[2].set_xlabel('Visual Word')
#         ax[2].set_ylabel('Frequency')
#         ax[2].set_title('Nearest Pos Train Histogram')

#         # Create a bar graph for the nearest negative training histogram
#         ax[3].bar(x_coords, vBoWNeg[nearest_neg_index], bar_width, color='r')
#         ax[3].set_xlabel('Visual Word')
#         ax[3].set_ylabel('Frequency')
#         ax[3].set_title('Nearest Neg Train Histogram')

#         # Save the image with the histograms and nearest neighbor histograms visualizations
#         output_image_path = os.path.join(inference_dir, os.path.basename(image_path))
#         plt.savefig(output_image_path)
#         plt.close()  # Close the figure to avoid multiple overlapping visualizations


if __name__ == '__main__':
    nameDirPos_train = 'data/data_bow/cars-training-pos'
    nameDirNeg_train = 'data/data_bow/cars-training-neg'
    nameDirPos_test = 'data/data_bow/cars-testing-pos'
    nameDirNeg_test = 'data/data_bow/cars-testing-neg'

    # HYPERPARAMETERS
    k = 15
    numiter = 50

    print('creating codebook ...')
    vCenters = create_codebook(nameDirPos_train, nameDirNeg_train, k, numiter)

    print('creating bow histograms (pos) ...')
    vBoWPos = create_bow_histograms(nameDirPos_train, vCenters)
    print('creating bow histograms (neg) ...')
    vBoWNeg = create_bow_histograms(nameDirNeg_train, vCenters)

    # test pos samples
    print('creating bow histograms for test set (pos) ...')
    vBoWPos_test = create_bow_histograms(nameDirPos_test, vCenters)  # [n_imgs, k]
    result_pos = 0
    print('testing pos samples ...')
    for i in range(vBoWPos_test.shape[0]):
        cur_label = bow_recognition_nearest(vBoWPos_test[i:(i+1)], vBoWPos, vBoWNeg)
        result_pos = result_pos + cur_label
    acc_pos = result_pos / vBoWPos_test.shape[0]
    print('test pos sample accuracy:', acc_pos)

    # test neg samples
    print('creating bow histograms for test set (neg) ...')
    vBoWNeg_test = create_bow_histograms(nameDirNeg_test, vCenters)  # [n_imgs, k]
    result_neg = 0
    print('testing neg samples ...')
    for i in range(vBoWNeg_test.shape[0]):
        cur_label = bow_recognition_nearest(vBoWNeg_test[i:(i + 1)], vBoWPos, vBoWNeg)
        result_neg = result_neg + cur_label
    acc_neg = 1 - result_neg / vBoWNeg_test.shape[0]
    print('test neg sample accuracy:', acc_neg)

    # # Save test samples with grid points
    # print('running inference for overlays (pos) ...')
    # save_images_with_overlays(nameDirPos_test, 'data/inference/pos', vBoWPos_test, vBoWPos, vBoWNeg)
    # print('running inference for overlays (neg) ...')
    # save_images_with_overlays(nameDirNeg_test, 'data/inference/neg', vBoWNeg_test, vBoWPos, vBoWNeg)