import mediapipe as mp
import cv2
import matplotlib.pyplot as plt 
from dtaidistance import dtw_visualisation as dtwvis
import time

from plot_data import *

def render_detections(video_path):
    """
    Processes a video to detect and render human poses using MediaPipe.
    Args:
        video_path (str): The path to the video file to be processed.
    Returns:
        None
    This function captures frames from the specified video file, processes each frame to detect human poses using 
    MediaPipe's Pose solution, and renders the detected landmarks on the frames. The processed frames are displayed 
    in a window. The function continues to process and display frames until the video ends or the user presses the 
    'q' key to exit.
    Note:
        - The function uses OpenCV for video capture and display.
        - The function uses MediaPipe for pose detection.
        - The video frames are resized to 1280x720 for display.
    """

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)

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
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(94, 218, 250), thickness=2, circle_radius=2) 
                                     )        

            image = cv2.resize(image, (1280, 720))
            cv2.imshow('Pose Estimation', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
def graph_paths(data, video_path, show_path=False):

    # Smooth the data
    smoothed_data = smooth_data(data, window_size=10)
    
    # Get the paths
    paths = get_paths(smoothed_data)
    path_arm = paths[0]
    path_fingers = paths[1]
    
    # Obtain the scores
    score_list = obtain_score(paths)
    
    # Record the start time
    start_time = time.time()
    
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

    # THIRD PLOT -> Text to show accuracy
    text_kwargs = dict(ha='center', va='center', fontsize=20, color='k')
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
    
    # Record the end time
    end_time = time.time()
    
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    fig.tight_layout()
    plt.show()
    return elapsed_time