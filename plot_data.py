from dtaidistance import dtw

def smooth_data(data, window_size):
    # Specify the columns to smooth
    columns_to_smooth = ['Arm Angle', 'Finger Distance']
    # Apply the rolling average to specific columns
    smoothed_data = data.copy()
    for column in columns_to_smooth:
        smoothed_data[column] = smoothed_data[column].rolling(window=window_size, min_periods=1).mean()
    return smoothed_data

def get_paths(smoothed_data):
    path_arm = dtw.warping_path(smoothed_data.loc[smoothed_data['Player'] == 'Pro', 'Arm Angle'].tolist(),
                                    smoothed_data.loc[smoothed_data['Player'] == 'User', 'Arm Angle'].tolist())
    path_fingers = dtw.warping_path(smoothed_data.loc[smoothed_data['Player'] == 'Pro', 'Finger Distance'].tolist(),
                                    smoothed_data.loc[smoothed_data['Player'] == 'User', 'Finger Distance'].tolist())
    return [path_arm, path_fingers]

def obtain_score(path_list):
    score_list = []
    for path in path_list:
        
        ## Remove all points that are mapped to 0
        path_non_zeros = []
        for point in path:
            if point[0] != 0 and point[1] != 0:
                path_non_zeros.append(point)

        ## Obtain distances for each point
        
        # Calculate the 1D distances and store them in a list
        distances = [abs(point[0] - point[1]) for point in path_non_zeros]
        
        # Calculate the average distance
        average_distance = sum(distances) / len(distances)
        
        # Normalize by dividing by the max distance in the list
        normalized_avg_distance = average_distance / max(distances)

        score_list.append(normalized_avg_distance)
    return score_list