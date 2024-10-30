import os
import cv2
import numpy as np

from src.AdityaMehta.Helpers.KeypointDetector import detect_keypoints, detect_keypoints_orb
from src.AdityaMehta.Helpers.KeypointMatcher import match_keypoints
from src.AdityaMehta.Helpers.Homography import ransac
from src.AdityaMehta.Helpers.Warper import warp_image

def calculate_final_image_size(left, right, H):
    left_corners = np.array([[0, 0, 1],
                                [left.shape[1] - 1, 0, 1],
                                [left.shape[1] - 1, left.shape[0] - 1, 1],
                                [0, left.shape[0] - 1, 1]])
    
    right_corners = np.array([[0, 0, 1],
                                [right.shape[1] - 1, 0, 1],
                                [right.shape[1] - 1, right.shape[0] - 1, 1],
                                [0, right.shape[0] - 1, 1]])
    
    transformed_right_corners = H @ right_corners.T
    transformed_right_corners /= transformed_right_corners[2, :]

    # Take min and max of x and y coordinates of all the corners
    max_x = int(max(np.max(left_corners[:, 0]), np.max(transformed_right_corners[0, :])))
    min_x = int(min(np.min(left_corners[:, 0]), np.min(transformed_right_corners[0, :])))
    max_y = int(max(np.max(left_corners[:, 1]), np.max(transformed_right_corners[1, :])))
    min_y = int(min(np.min(left_corners[:, 1]), np.min(transformed_right_corners[1, :])))

    return max_x, min_x, max_y, min_y

def distance_weighted_blend(left_overlap, right_overlap):
    # Convert overlaps to grayscale for mask creation
    left_gray = cv2.cvtColor(left_overlap, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_overlap, cv2.COLOR_BGR2GRAY)

    # Calculate masks for non-zero regions
    left_mask = (left_gray > 0).astype(np.uint8)
    right_mask = (right_gray > 0).astype(np.uint8)

    # Apply distance transform to create weighting matrices
    left_dist_transform = cv2.distanceTransform(left_mask, cv2.DIST_L2, 3)
    right_dist_transform = cv2.distanceTransform(right_mask, cv2.DIST_L2, 3)

    # Normalize to create weight maps
    left_weights = left_dist_transform / (left_dist_transform + right_dist_transform + 1e-6)
    right_weights = right_dist_transform / (left_dist_transform + right_dist_transform + 1e-6)

    # Blend the overlapping regions
    blended_overlap = (left_overlap * left_weights[..., None] + right_overlap * right_weights[..., None]).astype(np.uint8)

    return blended_overlap

def stitch2right(left, right, H):
    max_x, min_x, max_y, min_y = calculate_final_image_size(left, right, H)

    output_width = max_x - min_x
    output_height = max_y - min_y

    output_img = np.zeros((output_height + 5, output_width + 5, 3), dtype=left.dtype)

    left_offset_x = -min_x
    left_offset_y = -min_y
    output_img[left_offset_y:left_offset_y + left.shape[0], left_offset_x:left_offset_x + left.shape[1]] = left

    warped_right = warp_image(right, H, output_img.shape, -left_offset_x, -left_offset_y)

    overlap_x_start = max(0, left_offset_x)
    overlap_x_end = min(output_img.shape[1], left_offset_x + left.shape[1])

    left_overlap = output_img[:, overlap_x_start:overlap_x_end]
    right_overlap = warped_right[:, overlap_x_start:overlap_x_end]

    # Perform distance-based blending
    blended_overlap = distance_weighted_blend(left_overlap, right_overlap)

    mask = (left_overlap > 0) & (right_overlap > 0)
    output_img[:, overlap_x_start:overlap_x_end][mask] = blended_overlap[mask]
    output_img[(output_img == 0) & (warped_right > 0)] = warped_right[(output_img == 0) & (warped_right > 0)]

    return output_img, warped_right

def stitch2left(left, right, H):
    H_inv = np.linalg.inv(H)
    max_x, min_x, max_y, min_y = calculate_final_image_size(right, left, H_inv)

    output_width = max_x - min_x
    output_height = max_y - min_y

    output_img = np.zeros((output_height + 5, output_width + 5, 3), dtype=right.dtype)

    right_offset_x = -min_x
    right_offset_y = -min_y
    output_img[right_offset_y:right_offset_y + right.shape[0], right_offset_x:right_offset_x + right.shape[1]] = right

    warped_left = warp_image(left, H_inv, output_img.shape, -right_offset_x, -right_offset_y)

    # Define overlap region
    overlap_x_start = max(0, right_offset_x)
    overlap_x_end = min(output_img.shape[1], right_offset_x + right.shape[1])

    right_overlap = output_img[:, overlap_x_start:overlap_x_end]
    left_overlap = warped_left[:, overlap_x_start:overlap_x_end]

    # Perform distance-based blending
    blended_overlap = distance_weighted_blend(left_overlap, right_overlap)

    mask = (left_overlap > 0) & (right_overlap > 0)
    output_img[:, overlap_x_start:overlap_x_end][mask] = blended_overlap[mask]
    output_img[(output_img == 0) & (warped_left > 0)] = warped_left[(output_img == 0) & (warped_left > 0)]

    return output_img, warped_left

def main():
    img1 = cv2.imread('Images/STA_0031.jpg')
    img2 = cv2.imread('Images/STB_0032.jpg')

    print("Images loaded")

    kp1, desc1 = detect_keypoints(img1)
    kp2, desc2 = detect_keypoints(img2)
    keypoints = [kp1, kp2]
    matches = match_keypoints(desc1, desc2, 0.2)

    print("Keypoints detected and matched")

    correspondenceList = np.array([
        [keypoints[0][match.queryIdx].pt[0], keypoints[0][match.queryIdx].pt[1],
        keypoints[1][match.trainIdx].pt[0], keypoints[1][match.trainIdx].pt[1]]
        for match in matches
    ])

    H, _ = ransac(correspondenceList)

    print("Homography matrix estimated")

    output_img, warped_right = stitch2left(img1, img2, H)

    print("Images stitched")

    if not os.path.exists('_stitches'):
        os.makedirs('_stitches')

    cv2.imwrite('_stitches/stitched_image.jpg', output_img)
    cv2.imwrite('_stitches/warped_right.jpg', warped_right)

    print("Stitched image saved to '_stitches' folder.")

if __name__ == "__main__":
    main()
