import rasterio
import numpy as np
import cv2

# Function to normalize depth image
def normalize_depth(depth_img):
    return cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Read the MS image (4 bands)
with rasterio.open('ms_image.tif') as ms_src:
    ms_image = ms_src.read()  # Read all bands
    ms_profile = ms_src.profile  # Profile to later write merged image

# Read the depth image (1 band)
with rasterio.open('depth_image.tif') as depth_src:
    depth_image = depth_src.read(1)  # Read the first band (assume only one band)
    depth_profile = depth_src.profile

# Normalize depth image
normalized_depth = normalize_depth(depth_image)

# Resize depth to match MS image (in case they aren't the same size)
height, width = ms_image.shape[1], ms_image.shape[2]
resized_depth = cv2.resize(normalized_depth, (width, height))

# Find the alignment using ORB (Feature-based alignment)
def align_images(ms_image_band, depth_image):
    # Convert both images to grayscale
    ms_gray = cv2.normalize(ms_image_band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_gray = depth_image

    # Initialize ORB detector
    orb = cv2.ORB_create(5000)

    # Find keypoints and descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(ms_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(depth_gray, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract the matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find homography matrix
    homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Use homography to warp the depth image to align with the MS image
    aligned_depth = cv2.warpPerspective(depth_image, homography_matrix, (ms_gray.shape[1], ms_gray.shape[0]))

    return aligned_depth

# Align depth image with the first band of MS image
aligned_depth = align_images(ms_image[0], resized_depth)

# Stack aligned depth with MS image to make it 5-band image
merged_image = np.vstack([ms_image, aligned_depth[np.newaxis, :, :]])

# Update profile to write 5-band image
ms_profile.update(count=5)

# Write the 5-band image
with rasterio.open('merged_image_5bands.tif', 'w', **ms_profile) as dst:
    dst.write(merged_image)

print("Merged image saved as 'merged_image_5bands.tif'")
