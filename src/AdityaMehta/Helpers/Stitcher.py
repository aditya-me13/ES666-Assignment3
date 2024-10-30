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

def stitch2right(left, right, H):
    return stitch_images(left, right, H)

def stitch2left(left, right, H):
    # Invert the homography matrix to warp the left image onto the right image's plane
    H_inv = np.linalg.inv(H)
    max_x, min_x, max_y, min_y = calculate_final_image_size(right, left, H_inv)

    output_width = max_x - min_x
    output_height = max_y - min_y

    output_img = np.zeros((output_height + 5, output_width + 5, 3), dtype=right.dtype)

    # Adjust the position of the right image in the output image
    right_offset_x = -min_x
    right_offset_y = -min_y
    output_img[right_offset_y:right_offset_y + right.shape[0], right_offset_x:right_offset_x + right.shape[1]] = right

    # Warp the left image using the inverse homography
    warped_left = warp_image(left, H_inv, output_img.shape, -right_offset_x, -right_offset_y)

    # Define overlap region
    overlap_x_start = max(0, right_offset_x)
    overlap_x_end = min(output_img.shape[1], right_offset_x + right.shape[1])

    # Create linear blend mask
    blend_width = overlap_x_end - overlap_x_start
    blend_mask = np.linspace(0, 1, blend_width).reshape(1, blend_width, 1)

    # Get overlapping regions for blending
    right_overlap = output_img[:, overlap_x_start:overlap_x_end]
    left_overlap = warped_left[:, overlap_x_start:overlap_x_end]

    # Apply the blend only where both images have data (non-zero)
    mask = (left_overlap > 0) & (right_overlap > 0)
    blended = (right_overlap * (1 - blend_mask) + left_overlap * blend_mask).astype(right.dtype)

    # Combine the images
    output_img[:, overlap_x_start:overlap_x_end][mask] = blended[mask]
    output_img[(output_img == 0) & (warped_left > 0)] = warped_left[(output_img == 0) & (warped_left > 0)]

    return output_img, warped_left


def stitch_images(left, right, H):
    max_x, min_x, max_y, min_y = calculate_final_image_size(left, right, H)

    output_width = max_x - min_x
    output_height = max_y - min_y

    output_img = np.zeros((output_height + 5, output_width + 5, 3), dtype=left.dtype)

    # Adjust the position of the left image in the output image
    left_offset_x = -min_x
    left_offset_y = -min_y
    output_img[left_offset_y:left_offset_y + left.shape[0], left_offset_x:left_offset_x + left.shape[1]] = left

    # Warp the right image
    warped_right = warp_image(right, H, output_img.shape, -left_offset_x, -left_offset_y)

    # Define overlap region
    overlap_x_start = max(0, left_offset_x)  # Start of overlap in x
    overlap_x_end = min(output_img.shape[1], left_offset_x + left.shape[1])  # End of overlap in x

    # Create linear blend mask
    blend_width = overlap_x_end - overlap_x_start
    blend_mask = np.linspace(0, 1, blend_width).reshape(1, blend_width, 1)  # Shape for broadcasting with RGB

    # Get overlapping regions for blending
    left_overlap = output_img[:, overlap_x_start:overlap_x_end]
    right_overlap = warped_right[:, overlap_x_start:overlap_x_end]

    # Apply the blend only where both images have data (non-zero)
    mask = (left_overlap > 0) & (right_overlap > 0)
    blended = (left_overlap * (1 - blend_mask) + right_overlap * blend_mask).astype(left.dtype)

    # Combine the images
    output_img[:, overlap_x_start:overlap_x_end][mask] = blended[mask]
    output_img[(output_img == 0) & (warped_right > 0)] = warped_right[(output_img == 0) & (warped_right > 0)]

    return output_img, warped_right

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

    output_img, warped_right = stitch_images(img1, img2, H)

    print("Images stitched")

    if not os.path.exists('_stitches'):
        os.makedirs('_stitches')

    cv2.imwrite('_stitches/stitched_image.jpg', output_img)
    cv2.imwrite('_stitches/warped_right.jpg', warped_right)

    print("Stitched image saved to '_stitches' folder.")

if __name__ == "__main__":
    main()
