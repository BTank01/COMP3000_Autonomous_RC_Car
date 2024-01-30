import os
import time
import numpy as np
import cv2 as cv
import glob
import pickle
import matplotlib.pyplot as plt

execution_path = os.getcwd()
image1 = "Test Images\Test_image_40cm.jpg"
image2 = "Test Images\Test_image_40cm.jpg"


class VisualSLAMMap:
    def __init__(self):
        self.keyframes = []  # List to store keyframes
        self.points3D = []   # List to store 3D points
        self.connections = []  # List to store connections between frames

    def add_keyframe(self, frame, pose):
        # Add a keyframe to the map
        self.keyframes.append({'frame': frame, 'pose': pose, 'points': []})

    def add_3D_point(self, point):
        # Add a 3D point to the map
        self.points3D.append(point)

    def associate_keyframe_points(self, keyframe_idx, points_indices):
        # Associate 3D points with a specific keyframe
        self.keyframes[keyframe_idx]['points'] = points_indices

    def update_map(self, new_keyframe, new_3D_points, connections):
        # Update the map with a new keyframe and 3D points
        self.add_keyframe(new_keyframe['frame'], new_keyframe['pose'])
        for point in new_3D_points:
            self.add_3D_point(point)
        self.associate_keyframe_points(len(self.keyframes) - 1, list(range(len(self.points3D) - len(new_3D_points), len(self.points3D))))

        # Update connections between frames
        self.connections.extend(connections)

    def prune_map(self):
        # Prune redundant or less informative points or keyframes
        # This step is problem-specific and depends on the map representation and application requirements
        pass

    def visualize_map(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot camera poses
        for keyframe in self.keyframes:
            pose = keyframe['pose']
            ax.scatter(pose[0, 3], pose[1, 3], pose[2, 3], c='r', marker='o')

        # Plot 3D points
        points3D = np.array(self.points3D)
        ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], c='b', marker='.')

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('Visual SLAM Map')

        plt.show()


# Test Function
def floor_plan(*raw_images):
    images = []
    for image in raw_images:
        images.append(cv.imread(image))
    # Create Stitcher
    stitcher = cv.Stitcher_create()

    # Stitch images
    status, image = stitcher.stitch(images)

    if status == cv.Stitcher_OK:
        # cv.imshow('Panorama', result)
        # cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("Error during stitching")

    """     cv.rectangle(image, (100, 50), (300, 200), color=(0, 255, 0), thickness=2)  # Example rectangle

    # Example: Add text annotations
    cv.putText(image, 'Kitchen', (120, 180), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

    # Display or save the result
    cv.imshow('Floor Plan with Annotations', image)
    cv.waitKey(0)
    cv.destroyAllWindows() """

    # Define the source points (coordinates of the original image)
    src_points = np.float32([[0, 0], [image.shape[1], 0], [0, image.shape[0]], [image.shape[1], image.shape[0]]])

    # Define the destination points (coordinates for the top-down view)
    dst_points = np.float32([[0, 0], [image.shape[1], 0], [0, image.shape[0]*2], [image.shape[1], image.shape[0]*2]])

    # Calculate the perspective transformation matrix
    M = cv.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective transformation to obtain the top-down view
    top_down_view = cv.warpPerspective(image, M, (image.shape[1], image.shape[0]*2))

    # Display or save the top-down view
    cv.imshow('Top-Down View', top_down_view)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Misc Functions
def display_matching_features(scene1_img, scene2_img, scene1Keypoints, scene2Keypoints, matches):
    final_img = cv.drawMatches(scene1_img, scene1Keypoints, scene2_img, scene2Keypoints, matches[:20],None)
    final_img = cv.resize(final_img, (1000, 650))
    # Show the final image
    cv.imshow("Matches", final_img)
    cv.waitKey(60000)


# Visualize reconstruction
def Visualize_reconstruction(points_3D):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D points
    ax.scatter(points_3D[:, 0], points_3D[:, 1], points_3D[:, 2], marker='o', s=5, c='r', alpha=0.5)

    # Configure the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


# Main Functions
def take_calibration_images(image_count, delay):
    file_location = "Calibration Images"
    video_stream = cv.VideoCapture(0)

    for i in range(0, image_count):
        success, frame = video_stream.read()
        if not success:
            break
        else:
            cv.imshow("Video", frame)
            cv.imwrite(f"{file_location}/image_{i}.jpg", frame)
            time.sleep(delay)

    video_stream.release()
    cv.destroyAllWindows()


def calibrate_camera():
    # https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
    # https://www.geeksforgeeks.org/camera-calibration-with-python-opencv/

    # Define the dimensions of checkerboard
    CHECKERBOARD = (10, 10)

    # stop the iteration when specified 
    # accuracy, epsilon, is reached or 
    # specified number of iterations are completed. 
    criteria = (cv.TERM_CRITERIA_EPS + 
                cv.TERM_CRITERIA_MAX_ITER, 100, 0.001) 

    # Vector for 3D points
    threedpoints = [] 

    # Vector for 2D points
    twodpoints = [] 

    #  3D points real world coordinates
    objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Extracting path of individual image stored
    # in a given directory. Since no path is
    # specified, it will take current directory
    # jpg files alone
    images = glob.glob("Calibration Images/*.jpg")
    
    for filename in images:
        image = cv.imread(filename) 
        grayColor = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        # If desired number of corners are
        # found in the image then ret = true
        ret, corners = cv.findChessboardCorners( 
                        grayColor, CHECKERBOARD,
                        cv.CALIB_CB_ADAPTIVE_THRESH 
                        + cv.CALIB_CB_FAST_CHECK +
                        cv.CALIB_CB_NORMALIZE_IMAGE)
        print(f"File: {filename} | Ret: {ret} | Corners {corners}")
    
        # If desired number of corners can be detected then,
        # refine the pixel coordinates and display
        # them on the images of checker board
        if ret:
            threedpoints.append(objectp3d) 
    
            # Refining pixel coordinates 
            # for given 2d points. 
            corners2 = cv.cornerSubPix( 
                grayColor, corners, (11, 11), (-1, -1), criteria) 
    
            twodpoints.append(corners2) 
    
            # Draw and display the corners 
            image = cv.drawChessboardCorners(image, CHECKERBOARD, corners2, ret) 
    
        cv.imshow('img', image) 
        cv.waitKey(0) 
    
    cv.destroyAllWindows() 
    
    h, w = image.shape[:2] 
    
    
    # Perform camera calibration by 
    # passing the value of above found out 3D points (threedpoints) 
    # and its corresponding pixel coordinates of the 
    # detected corners (twodpoints) 
    ret, matrix, distortion, r_vecs, t_vecs = cv.calibrateCamera( 
        threedpoints, twodpoints, grayColor.shape[::-1], None, None) 

    # Displaying required output
    print("Camera matrix:")
    print(matrix)
    with open("Calibration Images/camera_calibration.pickle", "wb") as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(matrix, f, pickle.HIGHEST_PROTOCOL)

    print("\n Distortion coefficient:") 
    print(distortion)
    with open("Calibration Images/camera_distortion.pickle", "wb") as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(distortion, f, pickle.HIGHEST_PROTOCOL)

    print("\n Rotation Vectors:")
    print(r_vecs)

    print("\n Translation Vectors:")
    print(t_vecs)


def reconstruction(scene1, scene2, distance):
    with open("Calibration Images/camera_calibration.pickle", "rb") as pickle_file:
        camera_matrix = pickle.load(pickle_file)
    # print(camera_matrix)
    with open("Calibration Images/camera_distortion.pickle", "rb") as pickle_file:
        camera_distortion = pickle.load(pickle_file)
    # print(camera_distortion)
        
    scene1_img = cv.imread(scene1)
    scene2_img = cv.imread(scene2)

    # Find keypoints and extract descriptors - ORB
    orb = cv.ORB_create()
    scene1Keypoints, scene1Descriptors = orb.detectAndCompute(scene1_img, None)
    scene2Keypoints, scene2Descriptors = orb.detectAndCompute(scene2_img, None)

    # Match Features between images - BruteForce Hamming
    matcher = cv.BFMatcher()
    matches = matcher.match(scene1Descriptors, scene2Descriptors)

    # display_matching_features(scene1_img, scene2_img, scene1Keypoints, scene2Keypoints, matches)  # Display image with matching features

    # Find camera matrix
    left_points = []
    right_points = []

    for i in range(0, len(matches)):
        left_points.append(scene1Keypoints[matches[i].queryIdx].pt)
        right_points.append(scene2Keypoints[matches[i].trainIdx].pt)

    # Calculate the essential matrix
    essential_matrix, mask = cv.findEssentialMat(np.array(left_points), np.array(right_points), cameraMatrix=camera_matrix, method=cv.RANSAC, prob=0.95, threshold=1.0)

    # Calculate Rotation and translation matrices
    retval, R, t, mask = cv.recoverPose(essential_matrix, np.array(left_points), np.array(right_points), cameraMatrix=camera_matrix, mask=mask)

    # Projection matrices for both scenes
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, t))

    P1 = camera_matrix @ P1
    P2 = camera_matrix @ P2

    # Point Triangulation
    points_4D = cv.triangulatePoints(P1, P2, np.array(left_points).reshape(-1, 1, 2), np.array(right_points).reshape(-1, 1, 2))
    points_3D = points_4D / points_4D[3]  # Convert to Cartesian coordinates
    points_3D = points_3D[:3, :].T

    # Visualize reconstruction
    Visualize_reconstruction(points_3D)


def reconstruction_multi(*image_paths):
    # initialize map
    vmap = VisualSLAMMap()

    # Load all images into opencv
    images = []
    for image in image_paths:
        images.append(cv.imread(image, cv.IMREAD_GRAYSCALE))

    # Load calibration matrix pickle
    with open("Calibration Images/camera_calibration.pickle", "rb") as pickle_file:
        camera_matrix = pickle.load(pickle_file)
    with open("Calibration Images/camera_distortion.pickle", "rb") as pickle_file:
        camera_distortion = pickle.load(pickle_file)

    # Initialize detector and matcher
    orb = cv.ORB_create()
    matcher = cv.BFMatcher()

    # Hold keypoints and descriptors
    keypoints_arr = []
    descriptors_arr = []

    for image in images:
        keypoints, descriptors = orb.detectAndCompute(image, None)
        keypoints_arr.append(keypoints)
        descriptors_arr.append(descriptors)

    matches_arr = []
    for i in range(0, len(descriptors_arr)-1):
        matches = matcher.match(descriptors_arr[i], descriptors_arr[i+1])
        matches = sorted(matches, key=lambda x: x.distance)
        matches_arr.append(matches)
    best_matches = matches[:50]

    # Triangulation
    points_3d = []
    points_2d = []
    for i, matches in enumerate(matches_arr):
        left_keypoints = keypoints_arr[i]
        right_keypoints = keypoints_arr[i + 1]

        Left_points = np.float32([left_keypoints[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
        right_points = np.float32([right_keypoints[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

        essential_matrix, mask = cv.findEssentialMat(right_points, Left_points, cameraMatrix=camera_matrix, method=cv.RANSAC, prob=0.95, threshold=1.0)
        _, R, t, mask = cv.recoverPose(essential_matrix, right_points, Left_points)

        # Projection Matricies
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = np.hstack((R, t))

        P1 = camera_matrix @ P1
        P2 = camera_matrix @ P2

        # Triangulate 3D points
        points_4d_homogeneous = cv.triangulatePoints(P1, P2, Left_points.reshape(-1, 2).T, right_points.reshape(-1, 2).T)
        points_3d_homogeneous = points_4d_homogeneous / points_4d_homogeneous[3, :]
        points_3d.append(points_3d_homogeneous[:3, :].T)
        points_2d.append(right_points)

    # Merge 3D points from all images
    points_3d = np.vstack(points_3d)
    points_2d = np.vstack(points_2d)

    # Visualize_reconstruction(points_3d)
    vmap.update_map({'frame': images[-1], 'pose': np.hstack((R, t))}, points_3d, [])
    vmap.visualize_map()


# calibrate_camera()
# For test object distance from object 2 = 150cm
# take_calibration_images(3, 5)
# reconstruction("Test Images/image3.jpg", "Test Images/image4.jpg", 37)
reconstruction_multi("Test Images/image3.jpg", "Test Images/image4.jpg")
# floor_plan("Test Images/image3.jpg", "Test Images/image4.jpg")

    # Birds EYE View
    # # Transform 3d points to birds eye view
    # image = images[1]
    # cv.imshow("", image)
    # average_height = np.median(points_3d[:, 1])
    # transformation_matrix = np.array([[1, 0, 0], [0, 0.01, 0], [0, -1/average_height, 0]])
    
    # image_size = (image.shape[1], image.shape[0])
    # K = cv.getOptimalNewCameraMatrix(camera_matrix, camera_distortion, image_size, alpha=-1, newImgSize=image_size)
    # K = K[0]
    # _, R_inv = cv.invert(R)
    # _, K_inv = cv.invert(K)
    # homography = np.matmul(np.identity(3), np.matmul(K, np.matmul(R_inv, K_inv)))
    # homography[0][0] = homography[0][0] * 1
    # homography[0][1] = homography[0][1] * 1
    # homography[0][2] = homography[0][2] * 1

    # homography[1][0] = homography[1][0] * 1
    # homography[1][1] = homography[1][1] * 1
    # homography[1][2] = homography[1][2] - 10000

    # homography[2][0] = homography[2][0] * 1
    # homography[2][1] = homography[2][1] * 1
    # # Zoom
    # homography[2][2] = homography[2][2] * -0.0000001

    
    # # print(transformation_matrix)
    # print(homography)
    # birdseye_view = cv.warpPerspective(image, homography, (image.shape[1], image.shape[0]), flags=cv.WARP_INVERSE_MAP)
    # cv.imshow('Birdseye View', birdseye_view)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
