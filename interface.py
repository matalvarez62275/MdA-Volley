import mediapipe as mp
import cv2
import matplotlib.pyplot as plt 
from dtaidistance import dtw_visualisation as dtwvis
import time
import tkinter as tk
from tkinter import messagebox

from plot_data import *

def render_detections(video_path):
    """
    Processes a video to detect and render human poses using MediaPipe.
    Args:
        video_path (str): The path to the video file to be processed.
    Returns:
        None
    This function captures frames from the specified video file, processes each frame to detect 
    human poses using MediaPipe's Pose solution, and renders the detected landmarks on the frames. 
    The processed frames are displayed in a window. The function continues to process and display 
    frames until the video ends or the user presses the 'q' key to exit.
    Note:
        - The function uses OpenCV for video capture and display.
        - The function uses MediaPipe for pose detection.
        - The video frames are resized to 1280x720 for display.
    """

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the beginning of the video
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
                                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, 
                                                               circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(94, 218, 250), thickness=2, 
                                                               circle_radius=2) 
                                         )        

                image = cv2.resize(image, (1280, 720))
                cv2.namedWindow('Pose Estimation', cv2.WINDOW_NORMAL | cv2.WINDOW_AUTOSIZE)
                cv2.setWindowProperty('Pose Estimation', cv2.WND_PROP_TOPMOST, 1)
                cv2.imshow('Pose Estimation', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

    cap.release()
    cv2.destroyAllWindows()

def graph_paths(data, video_path, show_path=False):
    """
    Generates and displays a series of graphs comparing the arm angle and finger distance 
    between a professional and a user, using Dynamic Time Warping (DTW) visualization.
    Parameters:
    data (pd.DataFrame): The input data containing the arm angle and finger distance for both 
                         the professional and the user.
    video_path (str): The path to the video file, displayed in the title if show_path is True.
    show_path (bool): If True, the video path is included in the graph title. Default is False.
    Returns:
    float: The elapsed time in seconds for generating and displaying the graphs.
    """

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
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
    colors = ['#00FFFF', '#FF00FF']  # Robotic style colors
    data_labels = ['Ángulo del codo', 'Distancia entre índice y pulgar']

    # Main Title
    title_kwargs = dict(ha='center', fontsize=24, color='#00FFFF', fontweight='bold', fontname='Arial')
    extra_text = ""
    if show_path:
        extra_text += "\n" + video_path
    fig.suptitle('Comparación de datos' + extra_text, **title_kwargs)

    ## Each graph 
    # FIRST PLOT - Ángulo del codo
    arm_angles_pro = smoothed_data.loc[smoothed_data['Player'] == 'Pro', 'Arm Angle'].tolist()
    arm_angles_user = smoothed_data.loc[smoothed_data['Player'] == 'User', 'Arm Angle'].tolist()
    
    dtwvis.plot_warping(arm_angles_pro,
                        arm_angles_user,
                        path_arm, 
                        fig=fig, axs=[ax[0,0], ax[1,0]], 
                        warping_line_options={'linewidth': 0.5, 'color': '#FFD700', 'alpha': 0.7})  # Golden lines
    ax[0, 0].set(ylabel='Ángulo [°]')
    ax[1, 0].set(xlabel='Frames', ylabel='Ángulo [°]')
    ax[0, 0].set_title(data_labels[0], fontsize=16, color='#00FFFF', fontweight='bold')
    ax[0, 0].text(1, 1, 'Profesional', ha='right', va='top', fontsize=12, color=colors[0], transform=ax[0,0].transAxes)
    ax[1, 0].text(1, 1, 'Usuario', ha='right', va='top', fontsize=12, color=colors[1], transform=ax[1,0].transAxes)

    # SECOND PLOT - Distancia entre índice y pulgar
    dtwvis.plot_warping(smoothed_data.loc[smoothed_data['Player'] == 'Pro', 'Finger Distance'].tolist(),
                        smoothed_data.loc[smoothed_data['Player'] == 'User', 'Finger Distance'].tolist(),
                        path_fingers, 
                        fig=fig, axs=[ax[0,1], ax[1,1]], 
                        warping_line_options={'linewidth': 0.5, 'color': '#FFD700', 'alpha': 0.7})  # Golden lines
    ax[0, 1].set(ylabel='Distancia [m]')
    ax[1, 1].set(xlabel='Frames', ylabel='Distancia [m]')
    ax[0, 1].set_title(data_labels[1], fontsize=16, color='#00FFFF', fontweight='bold')
    ax[0, 1].text(1, 1, 'Profesional', ha='right', va='top', fontsize=12, color=colors[0], transform=ax[0,1].transAxes)
    ax[1, 1].text(1, 1, 'Usuario', ha='right', va='top', fontsize=12, color=colors[1], transform=ax[1,1].transAxes)

    # Show similarity scores
    text_kwargs = dict(ha='center', va='center', fontsize=18, color='#FFFFFF', fontweight='bold')
    text2show1 =  f"Similitud del ángulo del codo: {round(100*score_list[0],2)}%"
    text2show1 += f"\nSimilitud de la distancia entre dedos: {round(100*score_list[1],2)}%"
    text2show1 += f"\nSimilitud total: {round(100*sum(score_list)/len(score_list),2)}%"
    
    fig.text(0.5, 0.08, text2show1, **text_kwargs, fontname='Consolas')  # Similitud individual
    #fig.text(0.5, 0.05, text2show2, **text_kwargs, fontname='Consolas')  # Similitud total

    # Background and aesthetic tweaks
    fig.patch.set_facecolor('#2E2E2E')  
    for i in range(2):
        for j in range(2):
            ax[i, j].set_facecolor('#1C1C1C')  
            ax[i, j].tick_params(axis='x', colors='#00FFFF')  
            ax[i, j].tick_params(axis='y', colors='#00FFFF')
            ax[i, j].spines['top'].set_color('#00FFFF')
            ax[i, j].spines['right'].set_color('#00FFFF')
            ax[i, j].spines['bottom'].set_color('#00FFFF')
            ax[i, j].spines['left'].set_color('#00FFFF')
            
    # Record the end time
    end_time = time.time()
    
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    fig.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust layout to fit similarity text
    plt.show()
    
    # Check for arm angle alert condition
    for angle in arm_angles_user:
        if abs(angle - 90) >= 10:
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            messagebox.showwarning("Arm Angle Alert", f"Arm angle is {angle} degrees, which is {abs(angle - 90)} degrees away from 90!")
            root.destroy()
            break
        
    return elapsed_time