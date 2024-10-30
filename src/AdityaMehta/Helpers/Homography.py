"""
Courtsy: https://github.com/hughesj919/HomographyEstimation/tree/master
"""

import os
import cv2
import random
import numpy as np

from src.AdityaMehta.Helpers.KeypointDetector import detect_keypoints
from src.AdityaMehta.Helpers.KeypointMatcher import match_keypoints

def calculate_homography(correspondences):
    a_list = []
    for corr in correspondences:
        x1, y1, x2, y2 = corr[0], corr[1], corr[2], corr[3]
        a_list.append([-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2])
        a_list.append([0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2])
    a_matrix = np.array(a_list)

    _, _, vh = np.linalg.svd(a_matrix)
    h = vh[-1, :].reshape(3, 3)
    h /= h[2, 2]
    return h

def geometric_distance(correspondences, h):
    p1 = np.hstack((correspondences[:, :2], np.ones((correspondences.shape[0], 1))))
    p2 = np.hstack((correspondences[:, 2:], np.ones((correspondences.shape[0], 1))))
    projected_p2 = (h @ p1.T).T
    projected_p2 /= projected_p2[:, [2]]
    errors = np.linalg.norm(projected_p2[:, :2] - p2[:, :2], axis=1)
    return errors

def ransac(correspondences, threshold = 5.0, iterations=1000):
    best_h = None
    max_inliers = []

    for _ in range(iterations):
        sampled_corrs = correspondences[np.random.choice(correspondences.shape[0], 4, replace=False)]
        h = calculate_homography(sampled_corrs)
        distances = geometric_distance(correspondences, h)
        inliers = correspondences[distances < threshold]

        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            best_h = h

        if len(max_inliers) > len(correspondences) * threshold:
            break
    
    inverse_h = np.linalg.inv(best_h)
    
    return inverse_h, max_inliers

def main():
    estimation_thresh = 5.0
    print(f"Estimation Threshold: {estimation_thresh}")

    img1 = cv2.imread('Images/STA_0031.jpg')
    img2 = cv2.imread('Images/STB_0032.jpg')

    kp1, desc1 = detect_keypoints(img1)
    kp2, desc2 = detect_keypoints(img2)
    keypoints = [kp1, kp2]
    matches = match_keypoints(desc1, desc2)

    correspondenceList = np.array([
        [keypoints[0][match.queryIdx].pt[0], keypoints[0][match.queryIdx].pt[1],
        keypoints[1][match.trainIdx].pt[0], keypoints[1][match.trainIdx].pt[1]]
        for match in matches
    ])

    final_h, inliers = ransac(correspondenceList, estimation_thresh)
    print("Final homography: ", final_h)
    print("Final inliers count: ", len(inliers))

    inlier_indices = [i for i, corr in enumerate(correspondenceList) if any(np.array_equal(corr, inlier) for inlier in inliers)]
    inlier_matches = [matches[i] for i in inlier_indices]

    if not os.path.exists('_homography'):
        os.makedirs('_homography')

    match_img = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None)
    cv2.imwrite('_homography/InlierMatches.png', match_img)
    np.savetxt('_homography/homography.txt', final_h)

    print("Inlier matches saved to '_homography' folder.")

    
if __name__ == "__main__":
    main()
