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

    print("\n Rotation Vectors:")
    print(r_vecs)

    print("\n Translation Vectors:")
    print(t_vecs)


def reconstruction(scene1, scene2, distance):
    with open("Calibration Images/camera_calibration.pickle", "rb") as pickle_file:
        camera_matrix = pickle.load(pickle_file)
    # print(camera_matrix)
        
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


# calibrate_camera()
# For test object distance from object 2 = 150cm
# take_calibration_images(3, 5)
# reconstruction("Test Images/image1.jpg", "Test Images/image2.jpg", 37)
