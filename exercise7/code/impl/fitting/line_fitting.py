import numpy as np
from matplotlib import pyplot as plt
import random

np.random.seed(0)
random.seed(0)

def least_square(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return k, b

def num_inlier(x,y,k,b,n_samples,thres_dist):
	# TODO
	# compute the number of inliers and a mask that denotes the indices of inliers
    y_estimated = k * x + b
    distance = np.abs(y - y_estimated)
    inlier_mask = distance < thres_dist
    num = np.sum(inlier_mask)
    return num, inlier_mask

def ransac(x,y,iter,n_samples,thres_dist,num_subset):
	# TODO
	# ransac
    best_inliers = 0
    k_ransac = None
    b_ransac = None
    inlier_mask = None
    
    for _ in range(iter):
        # Randomly choose a subset of points
        indices = np.random.choice(n_samples, num_subset, replace=False)
        x_subset = x[indices]
        y_subset = y[indices]
        
        # Compute least squares on the subset
        k_subset, b_subset = least_square(x_subset, y_subset)
        
        # Count the inliers for this model
        num, mask = num_inlier(x, y, k_subset, b_subset, n_samples, thres_dist)
        
        # Update the best model if this one is better
        if num > best_inliers:
            best_inliers = num
            k_ransac = k_subset
            b_ransac = b_subset
            inlier_mask = mask

    return k_ransac, b_ransac, inlier_mask

def main():
	iter = 300
	thres_dist = 1
	n_samples = 500
	n_outliers = 50
	k_gt = 1
	b_gt = 10
	num_subset = 5
	x_gt = np.linspace(-10,10,n_samples)
	print(x_gt.shape)
	y_gt = k_gt*x_gt+b_gt
	# add noise
	x_noisy = x_gt+np.random.random(x_gt.shape)-0.5
	y_noisy = y_gt+np.random.random(y_gt.shape)-0.5
	# add outlier
	x_noisy[:n_outliers] = 8 + 10 * (np.random.random(n_outliers)-0.5)
	y_noisy[:n_outliers] = 1 + 2 * (np.random.random(n_outliers)-0.5)

	# least square
	k_ls, b_ls = least_square(x_noisy, y_noisy)
     
	k_ransac, b_ransac, inlier_mask = ransac(x_noisy, y_noisy, iter, n_samples, thres_dist, num_subset)
	outlier_mask = np.logical_not(inlier_mask)

	print("Estimated coefficients (true, linear regression, RANSAC):")
	print(k_gt, b_gt, k_ls, b_ls, k_ransac, b_ransac)

	line_x = np.arange(x_noisy.min(), x_noisy.max())
	line_y_ls = k_ls*line_x+b_ls
	line_y_ransac = k_ransac*line_x+b_ransac

	plt.scatter(
	    x_noisy[inlier_mask], y_noisy[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
	)
	plt.scatter(
	    x_noisy[outlier_mask], y_noisy[outlier_mask], color="gold", marker=".", label="Outliers"
	)
	plt.plot(line_x, line_y_ls, color="navy", linewidth=2, label="Linear regressor")
	plt.plot(
	    line_x,
	    line_y_ransac,
	    color="cornflowerblue",
	    linewidth=2,
	    label="RANSAC regressor",
	)
	plt.legend()
	plt.xlabel("Input")
	plt.ylabel("Response")
	plt.show()

if __name__ == '__main__':
	main()