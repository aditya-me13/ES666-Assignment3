import cv2
import os

# Using SIFT to detect keypoints
def detect_keypoints(image):
    # Create SIFT detector
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    
    return keypoints, descriptors

# Using ORB to detect keypoints
def detect_keypoints_orb(image):
    # Create ORB detector
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(img_gray, None)
    
    return keypoints, descriptors

def main():
    image = cv2.imread('Images/STA_0031.jpg')
    keypoints, descriptors = detect_keypoints(image)
    keypoints_orb, descriptors_orb = detect_keypoints_orb(image)
    # Display keypoints
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_with_keypoints_orb = cv2.drawKeypoints(image, keypoints_orb, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Save keypoints into a folder called "Keypoints"
    if not os.path.exists('_keypoints'):
        os.makedirs('_keypoints')

    cv2.imwrite('_keypoints/keypoints_sift.jpg', img_with_keypoints)
    cv2.imwrite('_keypoints/keypoints_orb.jpg', img_with_keypoints_orb)

    # Show the results

    # cv2.imshow('Keypoints (SIFT)', img_with_keypoints)
    # cv2.imshow('Keypoints (ORB)', img_with_keypoints_orb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print("Keypoints saved to '_keypoints' folder.")
    
if __name__ == "__main__":
    main()