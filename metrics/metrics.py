import numpy as np
import mediapipe as mp
import pandas as pd
import cv2
import os

def calculate_angle(a,b,c):
    
    """
    Calculate the angle between three points.
    Parameters:
    a (array-like): Coordinates of the first point.
    b (array-like): Coordinates of the second point (vertex of the angle).
    c (array-like): Coordinates of the third point.
    Returns:
    float: The angle in degrees between the three points.
    """
    
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

def calculate_distance(a,b):
    
    """
    Calculate the Euclidean distance between two points.
    Parameters:
    a (array-like): The first point, can be a list or numpy array.
    b (array-like): The second point, can be a list or numpy array.
    Returns:
    float: The Euclidean distance between point a and point b.
    """
    
    a = np.array(a)
    b = np.array(b)
    
    distance = np.linalg.norm(a-b)
    
    return distance

def metrics(video_path):
    
    """
    Analyzes a video to extract pose metrics using MediaPipe and OpenCV.
    Args:
        video_path (str): The path to the video file to be analyzed.
    Returns:
        None: The function saves the extracted metrics to a CSV file in the 'metrics' directory.
    The function performs the following steps:
    1. Initializes MediaPipe pose estimation and OpenCV video capture.
    2. Processes each frame of the video to detect pose landmarks.
    3. Calculates the arm angle and finger distance for each frame.
    4. Stores the calculated metrics over time.
    5. Draws the pose landmarks on the video frames and displays them.
    6. Saves the extracted metrics to a CSV file in a 'metrics' directory located in the same directory as the video file.
    Note:
        - The function assumes the presence of `calculate_angle` and `calculate_distance` functions for metric calculations.
        - The video display window can be closed by pressing the 'q' key.
    """
    
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)

    ## Variables for angle graphs
    arm_angle_time = []
    finger_distance_time = []
    time = []

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            try:
                frame = cap.read()[1]

                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except:
                break
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]
                right_thumb = [landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].y]

                # Calculate values
                arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                finger_distance = calculate_distance(right_index, right_thumb)

                # Save values
                arm_angle_time.append(arm_angle)
                finger_distance_time.append(finger_distance)

                if len(time) == 0:
                    time.append(0)
                else:
                    time.append(max(time)+1)

            except:
                pass

        cap.release()

    # Store Data for the subplots
    data_lists = [arm_angle_time, finger_distance_time]
    data_labels = ['Arm Angle', 'Finger Distance']
    dfW = pd.DataFrame(dict(zip(data_labels, data_lists)))
    
    # Get the directory path
    directory_path = os.path.join(os.path.dirname(os.path.dirname(video_path)), 'metrics')
    
    # Check if the directory exists, if not, create it
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Save the CSV file
    dfW.to_csv(os.path.join(directory_path, os.path.basename(video_path)[:-4] + '.csv'))