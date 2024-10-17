import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def moving_avg(data, window_size):
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='same')

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

# Video Capture source (If 0 -> computer camera)
cap = cv2.VideoCapture('Videos_Pros/OneServe2.mp4')

# Curl counter variables
counter = 0 
stage = None

## Variables for angle graphs
left_arm_angle_time = []
right_arm_angle_time = []
left_leg_angle_time = []
right_leg_angle_time = []
time = []

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            
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
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            #print(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z)

            # Calculate angle
            left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
            
            #Save angle
            left_arm_angle_time.append(left_arm_angle)
            right_arm_angle_time.append(right_arm_angle)
            left_leg_angle_time.append(left_leg_angle)
            right_leg_angle_time.append(right_leg_angle)
            if len(time) == 0:
                time.append(0)
            else:
                time.append(max(time)+1)
            
            # Curl counter logic
            if left_arm_angle > 160:
                stage = "down"
            if left_arm_angle < 30 and stage =='down':
                stage="up"
                counter +=1
                print(counter)
                       
        except:
            pass
        
        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # Rep data
        cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Stage data
        cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        image = cv2.resize(image, (1280, 720))
        cv2.imshow('Pose Estimation', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Store Data for the subplots
data_lists = [left_arm_angle_time, right_arm_angle_time, left_leg_angle_time, right_leg_angle_time]
data_labels = ['Left Arm Angle', 'Right Arm Angle', 'Left Leg Angle', 'Right Leg Angle']
dfW = pd.DataFrame(dict(zip(data_labels, data_lists))) # create pandas dataframe
dfW.to_csv('Data.csv')

# Create the first graph
plt.figure(figsize=(10, 6))
plt.plot(time, left_arm_angle_time, label='Left Arm Angle', color='blue')
plt.plot(time, right_arm_angle_time, label='Right Arm Angle', color='green')
plt.plot(time, left_leg_angle_time, label='Left Leg Angle', color='red')
plt.plot(time, right_leg_angle_time, label='Right Leg Angle', color='purple')
# Add labels and title
plt.xlabel('Time')
plt.ylabel('Angle')
plt.title('All Angles vs. Time')
# Add legend
plt.legend()
# Add grid
plt.grid(True)
# Show the first graph
plt.show()

# Create the second graph, smoothed
window_size = 10
plt.figure(figsize=(10, 6))
plt.plot(time, moving_avg(left_arm_angle_time, window_size), label='Left Arm Angle', color='blue')
plt.plot(time, moving_avg(right_arm_angle_time, window_size), label='Right Arm Angle', color='green')
plt.plot(time, moving_avg(left_leg_angle_time, window_size), label='Left Leg Angle', color='red')
plt.plot(time, moving_avg(right_leg_angle_time, window_size), label='Right Leg Angle', color='purple')
# Add labels and title
plt.xlabel('Time')
plt.ylabel('Angle')
plt.title('All Angles (smoothed) vs. Time')
# Add legend
plt.legend()
# Add grid
plt.grid(True)
# Show the first graph
plt.show()


# Create the third graph with a 2x2 layout
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
# Data for the subplots
data_lists = [left_arm_angle_time, right_arm_angle_time, left_leg_angle_time, right_leg_angle_time]
data_labels = ['Left Arm Angle', 'Right Arm Angle', 'Left Leg Angle', 'Right Leg Angle']
colors = ['blue', 'green', 'red', 'purple']
# Loop through the subplots and plot each data list
for i, ax in enumerate(axes.flatten()):
    ax.plot(time, data_lists[i], linestyle='--', color=colors[i], label=data_labels[i])
    ax.plot(time, moving_avg(data_lists[i], window_size), color=colors[i], label="Smoothed"+data_labels[i])
    ax.set_xlabel('Time')
    ax.set_ylabel('Angle')
    ax.set_title(data_labels[i])
    ax.grid(True)
# Adjust layout
plt.tight_layout()
# Show the second graph
plt.show()