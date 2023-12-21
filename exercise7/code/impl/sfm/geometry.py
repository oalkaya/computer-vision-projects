import numpy as np

from impl.dlt import BuildProjectionConstraintMatrix
from impl.util import MakeHomogeneous, HNormalize
from impl.sfm.corrs import GetPairMatches
# from impl.opt import ImageResiduals, OptimizeProjectionMatrix

# # Debug
# import matplotlib.pyplot as plt
# from impl.vis import Plot3DPoints, PlotCamera, PlotProjectedPoints


def EstimateEssentialMatrix(K, im1, im2, matches):
  # TODO

  # Obtain the keypoints from image 1 and image 2 using the matches
  kps1 = im1.kps[matches[:, 0]]
  kps2 = im2.kps[matches[:, 1]]

  # Transpose (M, 3) -> (3, M) to be able to calculate x_hat = K^-1 * x.
  kps1_homo = MakeHomogeneous(kps1.T)
  kps2_homo = MakeHomogeneous(kps2.T)

  # Normalize coordinates (to points on the normalized image plane), transpose back to (M,3)
  K_inv = np.linalg.inv(K)
  normalized_kps1 = (K_inv @ kps1_homo).T
  normalized_kps2 = (K_inv @ kps2_homo).T

  # Assemble constraint matrix as equation 2.1
  constraint_matrix = np.zeros((matches.shape[0], 9))
  for i in range(matches.shape[0]):
    # TODO
    x1 = normalized_kps1[i]
    x2 = normalized_kps2[i]
    A_i = np.kron(x1.T, x2.T) # [x'x, x'y, x', y'x, y'y, y', x, y, 1]
    constraint_matrix[i] = A_i.flatten()
  
  # Solve for the nullspace of the constraint matrix
  _, _, vh = np.linalg.svd(constraint_matrix)
  vectorized_E_hat = vh[-1,:]

  # TODO
  # Reshape the vectorized matrix to it's proper shape again
  E_hat = vectorized_E_hat.reshape(3, 3)

  # TODO
  # We need to fulfill the internal constraints of E
  # The first two singular values need to be equal, the third one zero.
  # Since E is up to scale, we can choose the two equal singluar values arbitrarily
  U, S, Vh = np.linalg.svd(E_hat) # S unused
  S_new = np.array([1, 1, 0])
  E = U @ np.diag(S_new) @ Vh

  # This is just a quick test that should tell you if your estimated matrix is not correct
  # It might fail if you estimated E in the other direction (i.e. kp2' * E * kp1)
  # You can adapt it to your assumptions.
  for i in range(matches.shape[0]):
    kp1 = normalized_kps1[i] # Changed assert statement to fit our array structure
    kp2 = normalized_kps2[i]

    assert(abs(kp1.transpose() @ E @ kp2) < 0.01) # 0.01 originally
    # print("Passed assert")

  return E


def DecomposeEssentialMatrix(E):

  u, s, vh = np.linalg.svd(E)

  # Determine the translation up to sign
  t_hat = u[:,-1]

  W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
  ])

  # Compute the two possible rotations
  R1 = u @ W @ vh
  R2 = u @ W.transpose() @ vh

  # Make sure the orthogonal matrices are proper rotations (Determinant should be 1)
  if np.linalg.det(R1) < 0:
    R1 *= -1

  if np.linalg.det(R2) < 0:
    R2 *= -1

  # Assemble the four possible solutions
  sols = [
    (R1, t_hat),
    (R2, t_hat),
    (R1, -t_hat),
    (R2, -t_hat)
  ]

  return sols

def TriangulatePoints(K, im1, im2, matches):
  R1, t1 = im1.Pose()
  R2, t2 = im2.Pose()
  P1 = K @ np.append(R1, np.expand_dims(t1, 1), 1)
  P2 = K @ np.append(R2, np.expand_dims(t2, 1), 1)

  # Ignore matches that already have a triangulated point
  new_matches = np.zeros((0, 2), dtype=int)

  num_matches = matches.shape[0]
  for i in range(num_matches):
    p3d_idx1 = im1.GetPoint3DIdx(matches[i, 0])
    p3d_idx2 = im2.GetPoint3DIdx(matches[i, 1])
    if p3d_idx1 == -1 and p3d_idx2 == -1:
      new_matches = np.append(new_matches, matches[[i]], 0)


  num_new_matches = new_matches.shape[0]
  print(f'Number of new matches {num_new_matches}')
  points3D = np.zeros((num_new_matches, 3))

  for i in range(num_new_matches):

    kp1 = im1.kps[new_matches[i, 0], :]
    kp2 = im2.kps[new_matches[i, 1], :]

    # H & Z Sec. 12.2
    A = np.array([
      kp1[0] * P1[2] - P1[0],
      kp1[1] * P1[2] - P1[1],
      kp2[0] * P2[2] - P2[0],
      kp2[1] * P2[2] - P2[1]
    ])

    _, _, vh = np.linalg.svd(A)
    homogeneous_point = vh[-1]
    points3D[i] = homogeneous_point[:-1] / homogeneous_point[-1]

  # We need to keep track of the correspondences between image points and 3D points
  im1_corrs = new_matches[:,0]
  im2_corrs = new_matches[:,1]

  # TODO
  # Filter points behind the cameras by transforming them into each camera space and checking the depth (Z)
  # Make sure to also remove the corresponding rows in `im1_corrs` and `im2_corrs`

  # Initialize lists to hold in-front points and their correspondences
  in_front_points = []
  in_front_im1_corrs = []
  in_front_im2_corrs = []

  for i in range(num_new_matches):
      point_3D = points3D[i]

      # Transform the point into the camera coordinate systems
      point_cam1 = R1 @ point_3D + t1
      point_cam2 = R2 @ point_3D + t2

      # Check if the point is in front of both cameras
      if point_cam1[2] > 0 and point_cam2[2] > 0:
          in_front_points.append(point_3D)
          in_front_im1_corrs.append(im1_corrs[i])
          in_front_im2_corrs.append(im2_corrs[i])

  # Convert lists to numpy arrays
  in_front_points = np.array(in_front_points)
  in_front_im1_corrs = np.array(in_front_im1_corrs)
  in_front_im2_corrs = np.array(in_front_im2_corrs)

  return in_front_points, in_front_im1_corrs, in_front_im2_corrs

def EstimateImagePose(points2D, points3D, K):  

  # TODO
  # We use points in the normalized image plane.
  # This removes the 'K' factor from the projection matrix.
  # We don't normalize the 3D points here to keep the code simpler.

  # Transpose (M, 3) -> (3, M) to be able to calculate x_hat = K^-1 * x.
  points2D_homo = MakeHomogeneous(points2D.T)

  # Normalize coordinates (to points on the normalized image plane), transpose back to (M,3)
  K_inv = np.linalg.inv(K)
  normalized_points2D = (K_inv @ points2D_homo).T

  constraint_matrix = BuildProjectionConstraintMatrix(normalized_points2D, points3D)

  # We don't use optimization here since we would need to make sure to only optimize on the se(3) manifold
  # (the manifold of proper 3D poses). This is a bit too complicated right now.
  # Just DLT should give good enough results for this dataset.

  # Solve for the nullspace
  _, _, vh = np.linalg.svd(constraint_matrix)
  P_vec = vh[-1,:]
  P = np.reshape(P_vec, (3, 4), order='C')

  # Make sure we have a proper rotation
  u, s, vh = np.linalg.svd(P[:,:3])
  R = u @ vh

  if np.linalg.det(R) < 0:
    R *= -1

  _, _, vh = np.linalg.svd(P)
  C = np.copy(vh[-1,:])

  t = -R @ (C[:3] / C[3])

  return R, t

def TriangulateImage(K, image_name, images, registered_images, matches):

  # TODO 
  # Loop over all registered images and triangulate new points with the new image.
  # Make sure to keep track of all new 2D-3D correspondences, also for the registered images

  image = images[image_name]
  points3D = np.zeros((0,3))
  # You can save the correspondences for each image in a dict and refer to the `local` new point indices here.
  # Afterwards you just add the index offset before adding the correspondences to the images.
  corrs = {}

  for reg_image_name in registered_images:
      reg_image = images[reg_image_name]
      pair_matches = GetPairMatches(image_name, reg_image_name, matches)

      if pair_matches.size == 0:
          continue

      # Triangulate points between the new image and the registered image.
      new_points3D, new_corrs_im1, new_corrs_im2 = TriangulatePoints(K, image, reg_image, pair_matches)

      if new_points3D.size > 0:
          # Append the new points to the points3D array.
          points3D = np.vstack((points3D, new_points3D))
          
          # Update the index offset for the newly added points.
          offset = points3D.shape[0] - new_points3D.shape[0]
          
          # Store correspondences for the new image.
          if image_name not in corrs:
              corrs[image_name] = []
          corrs[image_name].extend([(kp_idx, offset + i) for i, kp_idx in enumerate(new_corrs_im2)])
          
          # Store correspondences for the registered image.
          if reg_image_name not in corrs:
              corrs[reg_image_name] = []
          corrs[reg_image_name].extend([(kp_idx, offset + i) for i, kp_idx in enumerate(new_corrs_im1)])

  return points3D, corrs

