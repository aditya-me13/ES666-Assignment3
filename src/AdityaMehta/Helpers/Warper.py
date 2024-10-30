import os
import cv2
import numpy as np

from src.AdityaMehta.Helpers.KeypointDetector import detect_keypoints, detect_keypoints_orb
from src.AdityaMehta.Helpers.KeypointMatcher import match_keypoints
from src.AdityaMehta.Helpers.Homography import ransac

def warp_image(target_img, H, output_shape, offset_x=0, offset_y=0):
    
    output_height, output_width = output_shape[:2]

    # Create an output image filled with zeros (black)
    warped_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    inverse_H = np.linalg.inv(H)

    # Iterate over every pixel in the output image
    for y in range(output_height):
        for x in range(output_width):
            # Create homogeneous coordinates for the output pixel with offset
            pixel_homogeneous = np.array([x + offset_x, y + offset_y, 1])

            # Map the pixel back to the target image's coordinates using the inverse homography matrix
            src_coords = inverse_H @ pixel_homogeneous

            # Normalize the coordinates
            src_x = src_coords[0] / src_coords[2]
            src_y = src_coords[1] / src_coords[2]

            # Check if the mapped coordinates are within the bounds of the target image
            if 0 <= src_x < target_img.shape[1] and 0 <= src_y < target_img.shape[0]:
                # Assign the pixel value from the target image to the warped image
                warped_image[y, x] = target_img[int(src_y), int(src_x)]

    return warped_image


def main():
    img1 = cv2.imread('Images/STA_0031.jpg')
    img2 = cv2.imread('Images/STB_0032.jpg')

    print("Images loaded")

    kp1, desc1 = detect_keypoints(img1)
    kp2, desc2 = detect_keypoints(img2)
    keypoints = [kp1, kp2]
    matches = match_keypoints(desc1, desc2)

    print("Keypoints detected and matched")

    correspondenceList = np.array([
        [keypoints[0][match.queryIdx].pt[0], keypoints[0][match.queryIdx].pt[1],
        keypoints[1][match.trainIdx].pt[0], keypoints[1][match.trainIdx].pt[1]]
        for match in matches
    ])

    H, _ = ransac(correspondenceList)

    print("Homography matrix estimated")
    
    warped_image = warp_image(img1, H, img2.shape[:2])

    print("Images warped")

    # Display the warped image  

    # cv2.imshow('Combined Image', combined_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if not os.path.exists('_warped'):
        os.makedirs('_warped')

    cv2.imwrite('_warped/warpped_image.jpg', warped_image)

    print("Warpped image saved to '_warped' folder.")

if __name__ == "__main__":
    main()
