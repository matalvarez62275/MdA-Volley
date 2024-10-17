import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# Variables for angle graphs from CVS file
Variable_1 = []
Variable_2 = []
Variable_3 = []

Player_V1 = []
Player_V2 = []
Player_V3 = []

time = []
################################################################

## Low Pass filter #############################################
window_size = 10
def moving_avg(data, window_size):
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='same')
################################################################

## Read data for the subplots
data_lists = [Variable_1, Variable_2, Variable_3]
data_lists2 = [Player_V1, Player_V2, Player_V3]
data_labels = ['Right Arm Angle', 'Full Body Angle', 'Left Leg Angle']

# File to read #################################################
df = pd.read_csv('Jugador2.csv')
################################################################
time = df.iloc[:,0] # La cantidad de datos indica la cantidad de frames
for i in range(3):
    data_lists[i] = df.iloc[:,i+1]


# File to read 2 ###############################################
df2 = pd.read_csv('Jugador3.csv')
################################################################
time2 = df2.iloc[:,0] # La cantidad de datos indica la cantidad de frames
for i in range(3):
    data_lists2[i] = df2.iloc[:,i+1]


## Create the graphs with a 2x2 layout ########################
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
colors = ['blue', 'red']

# Main Title
title_kwargs = dict(ha='center', fontsize=20, color='k')
fig.suptitle('Compare Metrics', **title_kwargs)

## Each graph 
# FIRST PLOT
axes[0, 0].plot(time, moving_avg(data_lists[0], window_size), color=colors[0], label='Pro')
axes[0, 0].plot(time2, moving_avg(data_lists2[0], window_size), color=colors[1], label='Player')
axes[0, 0].set(xlabel='Frames', ylabel='Angle')
axes[0, 0].set_title(data_labels[0])
axes[0, 0].legend(loc="lower right")

# SECOND PLOT
axes[0, 1].plot(time, moving_avg(data_lists[1], window_size), color=colors[0], label='Pro')
axes[0, 1].plot(time2, moving_avg(data_lists2[1], window_size), color=colors[1], label='Player')
axes[0, 1].set(xlabel='Frames', ylabel='Angle')
axes[0, 1].set_title(data_labels[1])
axes[0, 1].legend(loc="lower right")

# THIRD PLOT
axes[1, 0].plot(time, moving_avg(data_lists[2], window_size), color=colors[0], label='Pro')
axes[1, 0].plot(time2, moving_avg(data_lists2[2], window_size), color=colors[1], label='Player')
axes[1, 0].set(xlabel='Frames', ylabel='Angle')
axes[1, 0].set_title(data_labels[2])
axes[1, 0].legend(loc="lower right")

# FOURTH PLOT -> Text to show accuracy
text_kwargs = dict(ha='center', va='center', fontsize=25, color='k')
acc = 5
text = "Global Accuracy: "
text2show = text + str(acc) + '%'
axes[1, 1].text(0.5, 0.5, text2show , **text_kwargs) #Understand how it works


plt.legend()
plt.tight_layout()
plt.show()