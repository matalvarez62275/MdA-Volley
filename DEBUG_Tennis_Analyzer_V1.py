import numpy as np
import cv2
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt 
from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance import dtw
import tkinter as tk
from tkinter import filedialog
import os
################################################################

################################################################
## Obtain Path to video ########################################
################################################################
def get_video_name():
    # Create a root window (hidden)
    root = tk.Tk()
    root.withdraw()

    # Open a file dialog
    file_path = filedialog.askopenfilename()

    # Check if a file was selected
    valid = False
    if file_path[-3:] == "mp4" or file_path[-3:] == "MOV":
        print(f"Selected file: {file_path}")
        valid = True
    else:
        print("Error: Incorrect File Type Selected")

    # Remember to destroy the root window
    root.destroy()
    return valid, file_path
################################################################

## Low Pass filter #############################################
window_size = 10  # Adjust this to your desired window size
def smooth_data(data, window_size):
    # Specify the columns to smooth
    columns_to_smooth = ['Right Arm Angle', 'Full Body Angle', 'Left Leg Angle']
    # Apply the rolling average to specific columns
    smoothed_data = data.copy()
    for column in columns_to_smooth:
        smoothed_data[column] = smoothed_data[column].rolling(window=window_size, min_periods=1).mean()
    return smoothed_data
################################################################
## Calculate angle between to XY points ########################
################################################################
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

################################################################
### Obtain player (pro or user) data in list format ############
################################################################
def get_player_data(csv_name):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_name)

    # Access each column and convert it to a Python list
    right_arm = df['Right Arm Angle'].tolist()
    full_body = df['Full Body Angle'].tolist()
    left_leg = df['Left Leg Angle'].tolist()
    time = [i for i in range(len(right_arm))]
    data = [time, right_arm, full_body, left_leg]
    return data

################################################################
### Obtain user video ##########################################
################################################################
def get_user_video(name, resize):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    # Video Capture source (If 0 -> computer camera)################
    cap = cv2.VideoCapture(name)
    ################################################################

    # Curl counter variables
    counter = 0 
    stage = None

    ## Variables for angle graphs
    right_arm_angle_time = []
    full_body_angle_time = []
    left_leg_angle_time = []
    time = []

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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

                # Calculate angle
                right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                full_body_angle = calculate_angle(right_wrist, right_hip, right_ankle)
                left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
                
                #Save angle
                right_arm_angle_time.append(right_arm_angle)
                full_body_angle_time.append(full_body_angle)
                left_leg_angle_time.append(left_leg_angle)

                if len(time) == 0:
                    time.append(0)
                else:
                    time.append(max(time)+1)
                        
            except:
                pass
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )        

            if resize:
                image = cv2.resize(image, (1280, 720))
            cv2.imshow('Pose Estimation', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # Store Data for the subplots
    data_lists = [right_arm_angle_time, full_body_angle_time, left_leg_angle_time]
    data_labels = ['Right Arm Angle', 'Full Body Angle', 'Left Leg Angle']
    dfW = pd.DataFrame(dict(zip(data_labels, data_lists))) # create pandas dataframe
    dfW.to_csv('User_Data.csv')

################################################################
### Get a joint dataframe from pro and user data in list form ##
################################################################
def get_dataframe(pro_data, user_data):

    # Create lists for your data
    tennist = ["Pro"]*len(pro_data[0]) + ["User"]*len(user_data[0])
    time = pro_data[0] + user_data[0]
    right_arm = pro_data[1] + user_data[1]
    full_body = pro_data[2] + user_data[2]
    left_leg = pro_data[3] + user_data[3]

    # Create a DataFrame by specifying the column names and their corresponding lists
    data = {
        'Tennist': tennist,
        'Time': time,
        'Right Arm Angle': right_arm,
        'Full Body Angle': full_body,
        'Left Leg Angle': left_leg
    }

    df = pd.DataFrame(data)

    # Print the DataFrame
    return df
#######################################################################
#######################################################################

################################################################
### Get Dynamic Time Warping list for each component          ##
################################################################
def get_paths(smoothed_data):
    path_right_arm = dtw.warping_path(smoothed_data.loc[smoothed_data['Tennist'] == 'Pro', 'Right Arm Angle'].tolist(),
                                    smoothed_data.loc[smoothed_data['Tennist'] == 'User', 'Right Arm Angle'].tolist())
    path_left_leg = dtw.warping_path(smoothed_data.loc[smoothed_data['Tennist'] == 'Pro', 'Left Leg Angle'].tolist(),
                                    smoothed_data.loc[smoothed_data['Tennist'] == 'User', 'Left Leg Angle'].tolist())
    path_full_body = dtw.warping_path(smoothed_data.loc[smoothed_data['Tennist'] == 'Pro', 'Full Body Angle'].tolist(),
                                    smoothed_data.loc[smoothed_data['Tennist'] == 'User', 'Full Body Angle'].tolist())
    return [path_right_arm, path_left_leg, path_full_body]
#######################################################################
#######################################################################

#########################################################################
###  Obtain score by normalizing distance of movements between tennists##
#########################################################################
def obtain_score(path_list):
    score_list = []
    for path in path_list:
        #First remove all points that are mapped to 0
        path_non_zeros = []
        for point in path:
            if point[0] != 0 and point[1] != 0:
                path_non_zeros.append(point)

        #Now obtain distances for each point
        # Remember that point[0] is the frame from PRO video, and point[1] is the corresponding point in USER video
        # Calculate the 1D distances and store them in a list
        distances = [abs(point[0] - point[1]) for point in path_non_zeros]
        # Calculate the average distance
        average_distance = sum(distances) / len(distances)
        # Normalize by dividing by the max distance in the list
        normalized_avg_distance = average_distance / max(distances)

        #print("1D Distances between points:", distances)
        #print("Average 1D Distance:", average_distance)
        #print("Normalized score = ", normalized_avg_distance)
        #print("")
        score_list.append(normalized_avg_distance)
    return score_list
#######################################################################
#######################################################################

#################################################################
###  Plot graphs for DTW between tennist and print final score ##
#################################################################
def graph_paths(smoothed_data, path_right_arm, path_left_leg, path_full_body, video_path, show_path=False, save_fig=False):

    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(10, 6))
    colors = ['blue', 'red']
    data_labels = ['Right Arm Angle', 'Full Body Angle', 'Left Leg Angle']

    # Main Title
    title_kwargs = dict(ha='center', fontsize=20, color='k')
    extra_text = ""
    if show_path:
        extra_text += "\n" + "File: " + os.path.basename(video_path)
    fig.suptitle('Compare Metrics' + extra_text, **title_kwargs)

    ## Each graph 
    # FIRST PLOT
    dtwvis.plot_warping(smoothed_data.loc[smoothed_data['Tennist'] == 'Pro', 'Right Arm Angle'].tolist(),
                        smoothed_data.loc[smoothed_data['Tennist'] == 'User', 'Right Arm Angle'].tolist(),
                        path_right_arm, 
                        fig=fig, axs=[ax[0,0], ax[1,0]], warping_line_options={'linewidth': 0.5, 'color': 'orange', 'alpha': 0.5})
    ax[0, 0].set(ylabel='Angle [°]')
    ax[1, 0].set(xlabel='Frames', ylabel='Angle [°]')
    ax[0, 0].set_title('Comparison between two tennist - ' + data_labels[0])
    ax[0, 0].text(1, 1, 'PRO', ha='right', va='top', fontsize=12, color=colors[0], transform=ax[0,0].transAxes)
    ax[1, 0].text(1, 1, 'USER', ha='right', va='top', fontsize=12, color=colors[1], transform=ax[1,0].transAxes)

    # SECOND PLOT
    dtwvis.plot_warping(smoothed_data.loc[smoothed_data['Tennist'] == 'Pro', 'Left Leg Angle'].tolist(),
                        smoothed_data.loc[smoothed_data['Tennist'] == 'User', 'Left Leg Angle'].tolist(),
                        path_left_leg, 
                        fig=fig, axs=[ax[0,1], ax[1,1]], warping_line_options={'linewidth': 0.5, 'color': 'orange', 'alpha': 0.5})
    ax[0, 1].set(ylabel='Angle [°]')
    ax[1, 1].set(xlabel='Frames', ylabel='Angle [°]')
    ax[0, 1].set_title('Comparison between two tennist - ' + data_labels[1])
    ax[0, 1].text(1, 1, 'PRO', ha='right', va='top', fontsize=12, color=colors[0], transform=ax[0,1].transAxes)
    ax[1, 1].text(1, 1, 'USER', ha='right', va='top', fontsize=12, color=colors[1], transform=ax[1,1].transAxes)

    # THIRD PLOT
    dtwvis.plot_warping(smoothed_data.loc[smoothed_data['Tennist'] == 'Pro', 'Full Body Angle'].tolist(),
                        smoothed_data.loc[smoothed_data['Tennist'] == 'User', 'Full Body Angle'].tolist(),
                        path_full_body, 
                        fig=fig, axs=[ax[2,0], ax[3,0]], warping_line_options={'linewidth': 0.5, 'color': 'orange', 'alpha': 0.5})
    ax[2, 0].set(ylabel='Angle [°]')
    ax[3, 0].set(xlabel='Frames', ylabel='Angle [°]')
    ax[2, 0].set_title('Comparison between two tennist - ' + data_labels[2])
    ax[2, 0].text(1, 1, 'PRO', ha='right', va='top', fontsize=12, color=colors[0], transform=ax[2,0].transAxes)
    ax[3, 0].text(1, 1, 'USER', ha='right', va='top', fontsize=12, color=colors[1], transform=ax[3,0].transAxes)

    # FOURTH PLOT -> Text to show accuracy
    text_kwargs = dict(ha='center', va='center', fontsize=20, color='k')
    acc = 5
    text2show1 =  "Right Arm Accuracy: " + str(round(100*score_list[0],2)) + '%'
    text2show1 += "\n" + "Left Leg Accuracy: " + str(round(100*score_list[1],2)) + '%'
    text2show1 += "\n" + "Full Body Accuracy: " + str(round(100*score_list[2],2)) + '%'
    text2show2 = "Global Accuracy: " + str(round(100*sum(score_list)/len(score_list),2)) + '%'
    ax[2, 1].text(0.5, 0, text2show1 , **text_kwargs) #Understand how it works
    ax[3, 1].text(0.5, 0, text2show2 , **text_kwargs) #Understand how it works

    #Hide spines and ticks
    ax[2, 1].set_xticks([])
    ax[2, 1].set_yticks([])
    ax[2, 1].spines['top'].set_visible(False)
    ax[2, 1].spines['right'].set_visible(False)
    ax[2, 1].spines['bottom'].set_visible(False)
    ax[2, 1].spines['left'].set_visible(False)
    ax[3, 1].set_xticks([])
    ax[3, 1].set_yticks([])
    ax[3, 1].spines['top'].set_visible(False)
    ax[3, 1].spines['right'].set_visible(False)
    ax[3, 1].spines['bottom'].set_visible(False)
    ax[3, 1].spines['left'].set_visible(False)

    fig.tight_layout()
    if save_fig:
        # Save the figure as a PNG with a different name
        filename = os.path.basename(video_path) 
        plt.savefig(filename[:-4] + ".png")
        
        # Close the current figure to release resources
        plt.close()
        print("Saved fig as ", filename[:-4] + ".png")
    else:
        plt.show()
    return
#######################################################################
#######################################################################

#######################################################################
###########################      MAIN      ############################
#######################################################################

def list_files_in_folder(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths



#valid = False

#valid, video_path = get_video_name()
folder_path = "C:/Users/HP/Documents/GitHub/TP-G1_MdA/Videos_Equipo/Right"
file_paths = list_files_in_folder(folder_path)
for video_path in file_paths:
    try:
        print("Now Scoring: ", os.path.basename(video_path))
        get_user_video(video_path, resize=True)
        pro_data = get_player_data('PRO_Right_Side.csv')
        user_data = get_player_data('User_Data.csv')
        data = get_dataframe(pro_data, user_data)
        smoothed_data = smooth_data(data, window_size=10)
        paths = get_paths(smoothed_data)
        score_list = obtain_score(get_paths(smoothed_data))
        graph_paths(smoothed_data, paths[0], paths[1], paths[2], video_path, show_path=True, save_fig=True)
    except:
        print("Error! Could not score: ", os.path.basename(video_path))
        pass

print("Finished scoring :)")