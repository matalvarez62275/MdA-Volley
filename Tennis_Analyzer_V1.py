import numpy as np
import cv2
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt 
from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance import dtw

from miscellaneous.handle_files import *
from metrics.func import *


## Low Pass filter #############################################
window_size = 10  # Adjust this to your desired window size
def smooth_data(data, window_size):
    # Specify the columns to smooth
    columns_to_smooth = ['Arm Angle', 'Finger Distance']
    # Apply the rolling average to specific columns
    smoothed_data = data.copy()
    for column in columns_to_smooth:
        smoothed_data[column] = smoothed_data[column].rolling(window=window_size, min_periods=1).mean()
    return smoothed_data

### Obtain player (pro or user) data in list format ############
def get_player_data(csv_name):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_name)

    # Access each column and convert it to a Python list
    arm = df['Arm Angle'].tolist()
    fingers = df['Finger Distance'].tolist()
    time = [i for i in range(len(arm))]
    data = [time, arm, fingers]
    return data

################################################################

################################################################
### Get a joint dataframe from pro and user data in list form ##
################################################################
def get_dataframe(pro_data, user_data):

    # Create lists for your data
    player = ["Pro"]*len(pro_data[0]) + ["User"]*len(user_data[0])
    time = pro_data[0] + user_data[0]
    arm = pro_data[1] + user_data[1]
    fingers = pro_data[2] + user_data[2]

    # Create a DataFrame by specifying the column names and their corresponding lists
    data = {
        'Player': player,
        'Time': time,
        'Arm Angle': arm,
        'Finger Distance': fingers
    }

    df = pd.DataFrame(data)

    # Print the DataFrame
    return df
#######################################################################

################################################################
### Get Dynamic Time Warping list for each component          ##
################################################################
def get_paths(smoothed_data):
    path_arm = dtw.warping_path(smoothed_data.loc[smoothed_data['Player'] == 'Pro', 'Arm Angle'].tolist(),
                                    smoothed_data.loc[smoothed_data['Player'] == 'User', 'Arm Angle'].tolist())
    path_fingers = dtw.warping_path(smoothed_data.loc[smoothed_data['Player'] == 'Pro', 'Finger Distance'].tolist(),
                                    smoothed_data.loc[smoothed_data['Player'] == 'User', 'Finger Distance'].tolist())
    return [path_arm, path_fingers]
################################################################

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

#################################################################
###  Plot graphs for DTW between tennist and print final score ##
#################################################################
def graph_paths(smoothed_data, path_arm, path_fingers, video_path, show_path=False):

    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(10, 6))
    colors = ['blue', 'red']
    data_labels = ['Arm Angle', 'Finger Distance']

    # Main Title
    title_kwargs = dict(ha='center', fontsize=20, color='k')
    extra_text = ""
    if show_path:
        extra_text += "\n" + video_path
    fig.suptitle('Compare Metrics' + extra_text, **title_kwargs)

    ## Each graph 
    # FIRST PLOT
    dtwvis.plot_warping(smoothed_data.loc[smoothed_data['Player'] == 'Pro', 'Arm Angle'].tolist(),
                        smoothed_data.loc[smoothed_data['Player'] == 'User', 'Arm Angle'].tolist(),
                        path_arm, 
                        fig=fig, axs=[ax[0,0], ax[1,0]], warping_line_options={'linewidth': 0.5, 'color': 'orange', 'alpha': 0.5})
    ax[0, 0].set(ylabel='Angle [°]')
    ax[1, 0].set(xlabel='Frames', ylabel='Angle [°]')
    ax[0, 0].set_title('Comparison between two players - ' + data_labels[0])
    ax[0, 0].text(1, 1, 'PRO', ha='right', va='top', fontsize=12, color=colors[0], transform=ax[0,0].transAxes)
    ax[1, 0].text(1, 1, 'USER', ha='right', va='top', fontsize=12, color=colors[1], transform=ax[1,0].transAxes)

    # SECOND PLOT
    dtwvis.plot_warping(smoothed_data.loc[smoothed_data['Player'] == 'Pro', 'Finger Distance'].tolist(),
                        smoothed_data.loc[smoothed_data['Player'] == 'User', 'Finger Distance'].tolist(),
                        path_fingers, 
                        fig=fig, axs=[ax[0,1], ax[1,1]], warping_line_options={'linewidth': 0.5, 'color': 'orange', 'alpha': 0.5})
    ax[0, 1].set(ylabel='Distance [m]')
    ax[1, 1].set(xlabel='Frames', ylabel='Distance [m]')
    ax[0, 1].set_title('Comparison between two players - ' + data_labels[1])
    ax[0, 1].text(1, 1, 'PRO', ha='right', va='top', fontsize=12, color=colors[0], transform=ax[0,1].transAxes)
    ax[1, 1].text(1, 1, 'USER', ha='right', va='top', fontsize=12, color=colors[1], transform=ax[1,1].transAxes)

    # FOURTH PLOT -> Text to show accuracy
    text_kwargs = dict(ha='center', va='center', fontsize=20, color='k')
    acc = 5
    text2show1 =  "Arm Accuracy: " + str(round(100*score_list[0],2)) + '%'
    text2show1 += "\n" + "Fingers Accuracy: " + str(round(100*score_list[1],2)) + '%'
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
    plt.show()
    return
#######################################################################

#######################################################################
###########################      MAIN      ############################
#######################################################################
valid = False

valid, video_path = get_video_name()
if valid:
    metrics(video_path)                                 
    pro_data = get_player_data('PRO/metrics/PRO1-right.csv')
    user_data = get_player_data('USER/metrics/10mv1.csv')
    data = get_dataframe(pro_data, user_data)
    smoothed_data = smooth_data(data, window_size=10)
    paths = get_paths(smoothed_data)
    score_list = obtain_score(get_paths(smoothed_data))
    graph_paths(smoothed_data, paths[0], paths[1], video_path, show_path=True)