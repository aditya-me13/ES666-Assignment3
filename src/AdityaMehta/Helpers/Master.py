import os
import cv2
import numpy as np

from src.AdityaMehta.Helpers.KeypointDetector import detect_keypoints
from src.AdityaMehta.Helpers.KeypointMatcher import match_keypoints
from src.AdityaMehta.Helpers.Homography import ransac
from src.AdityaMehta.Helpers.Stitcher import stitch2left, stitch2right

SCALE = 70 # Update Accordingly here

def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def stitch_multiple_images(folder_path):
    # Load all images from the folder and sort them by filename
    image_filenames = sorted([f for f in os.listdir(folder_path) if f.endswith(('.JPG', '.PNG', '.jpg', '.png'))])
    images = [cv2.imread(os.path.join(folder_path, f)) for f in image_filenames]

    print("Images loaded:", image_filenames)

    if len(images) < 2:
        print("Not enough images to stitch.")
        return None, []

    # Resize all images
    images = [resize_image(img, SCALE) for img in images]

    # Determine the middle index
    mid_index = len(images) // 2
    stitched_image = images[mid_index]
    print("Starting stitching from the middle image:", image_filenames[mid_index])

    # Initialize a list to store homography matrices
    homography_matrices = []

    # Stitch images to the right of the middle image
    for i in range(mid_index + 1, len(images)):
        img1 = stitched_image
        img2 = images[i]

        kp1, desc1 = detect_keypoints(img1)
        kp2, desc2 = detect_keypoints(img2)
        
        matches = match_keypoints(desc1, desc2, 0.2)
        print(f"Keypoints detected and matched for images: {image_filenames[i-1]} and {image_filenames[i]}")

        correspondenceList = np.array([
            [kp1[match.queryIdx].pt[0], kp1[match.queryIdx].pt[1],
            kp2[match.trainIdx].pt[0], kp2[match.trainIdx].pt[1]]
            for match in matches
        ])

        H, _ = ransac(correspondenceList)
        homography_matrices.append(H)
        print(f"Homography matrix estimated for images: {image_filenames[i-1]} and {image_filenames[i]}")

        # Stitch the current stitched image with the next image
        stitched_image, _ = stitch2right(img1, img2, H)

    # Stitch images to the left of the middle image
    for i in range(mid_index - 1, -1, -1):
        img1 = images[i]
        img2 = stitched_image

        kp1, desc1 = detect_keypoints(img1)
        kp2, desc2 = detect_keypoints(img2)

        matches = match_keypoints(desc1, desc2, 0.2)
        print(f"Keypoints detected and matched for images: {image_filenames[i]} and {image_filenames[i+1]}")

        correspondenceList = np.array([
            [kp1[match.queryIdx].pt[0], kp1[match.queryIdx].pt[1],
            kp2[match.trainIdx].pt[0], kp2[match.trainIdx].pt[1]]
            for match in matches
        ])

        H, _ = ransac(correspondenceList)
        homography_matrices.append(H)
        print(f"Homography matrix estimated for images: {image_filenames[i]} and {image_filenames[i+1]}")

        # Stitch the previous image to the left of the stitched image
        stitched_image, _ = stitch2left(img1, img2, H)

    # Return the final stitched image and list of homography matrices
    return stitched_image, homography_matrices

def main():
    stitched_image, homographies = stitch_multiple_images('Images')
    if stitched_image is not None:
        print("Stitching complete. Homography matrices:")
        for i, H in enumerate(homographies):
            print(f"Homography {i+1}:\n{H}")

        cv2.imshow("Stitched Image", stitched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
