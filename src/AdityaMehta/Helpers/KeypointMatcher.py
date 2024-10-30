import os
import cv2
import numpy as np

from src.AdityaMehta.Helpers.KeypointDetector import detect_keypoints, detect_keypoints_orb

def match_keypoints(descriptors1, descriptors2, n = 0.15):
    """
    Match keypoints between two sets of descriptors using BFMatcher.
    
    :param descriptors1: Descriptors of the first image
    :param descriptors2: Descriptors of the second image
    :return: Matches between the two images
    """
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    top_n = int(len(matches) * n)
    return matches[:top_n]

def match_keypoints_all(descriptors1, descriptors2):
    """
    Match keypoints between two sets of descriptors using BFMatcher.
    
    :param descriptors1: Descriptors of the first image
    :param descriptors2: Descriptors of the second image
    :return: Matches between the two images
    """
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    return matches

def main():
    image1 = cv2.imread('Images/STA_0031.jpg')
    image2 = cv2.imread('Images/STB_0032.jpg')

    # using SIFT to detect keypoints
    keypoints1, descriptors1 = detect_keypoints(image1)
    keypoints2, descriptors2 = detect_keypoints(image2)

    # using ORB to detect keypoints
    keypoints1_orb, descriptors1_orb = detect_keypoints_orb(image1)
    keypoints2_orb, descriptors2_orb = detect_keypoints_orb(image2)

    matches = match_keypoints(descriptors1, descriptors2)
    matches_orb = match_keypoints(descriptors1_orb, descriptors2_orb)
    
    # Draw matches
    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches_orb = cv2.drawMatches(image1, keypoints1_orb, image2, keypoints2_orb, matches_orb[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Save the image in a folder names 'matches'
    if not os.path.exists('_matches'):
        os.makedirs('_matches')

    cv2.imwrite('_matches/matches_sift.jpg', img_matches)
    cv2.imwrite('_matches/matches_orb.jpg', img_matches_orb)

    # Show the result

    # cv2.imshow('Matches SIFT', img_matches)
    # cv2.imshow('Matches ORB', img_matches_orb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print("Matches saved to '_matches' folder.")

if __name__ == "__main__":
    main()

