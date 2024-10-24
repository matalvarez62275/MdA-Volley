from dtaidistance import dtw

def smooth_data(data, window_size):
    """
    Smooths specified columns in the given DataFrame using a rolling average.

    Parameters:
    data (pandas.DataFrame): The input data containing the columns to be smoothed.
    window_size (int): The size of the rolling window to use for smoothing.

    Returns:
    pandas.DataFrame: A new DataFrame with the specified columns smoothed using a rolling average.
    """
    # Specify the columns to smooth
    columns_to_smooth = ['Arm Angle', 'Finger Distance']
    # Apply the rolling average to specific columns
    smoothed_data = data.copy()
    for column in columns_to_smooth:
        smoothed_data[column] = smoothed_data[column].rolling(window=window_size, min_periods=1).mean()
    return smoothed_data

def get_paths(smoothed_data):
    """
    Computes the Dynamic Time Warping (DTW) paths for 'Arm Angle' and 'Finger Distance' between 
    'Pro' and 'User' players from the provided smoothed data.

    Parameters:
    smoothed_data (pd.DataFrame): A pandas DataFrame containing the smoothed data with columns 
                                  'Player', 'Arm Angle', and 'Finger Distance'.

    Returns:
    list: A list containing two DTW paths:
          - path_arm: DTW path for 'Arm Angle' between 'Pro' and 'User'.
          - path_fingers: DTW path for 'Finger Distance' between 'Pro' and 'User'.
    """
    path_arm = dtw.warping_path(smoothed_data.loc[smoothed_data['Player'] == 'Pro', 'Arm Angle'].tolist(),
                                    smoothed_data.loc[smoothed_data['Player'] == 'User', 'Arm Angle'].tolist())
    path_fingers = dtw.warping_path(smoothed_data.loc[smoothed_data['Player'] == 'Pro', 'Finger Distance'].tolist(),
                                    smoothed_data.loc[smoothed_data['Player'] == 'User', 'Finger Distance'].tolist())
    return [path_arm, path_fingers]

def obtain_score(path_list):
    """
    Calculate a normalized average distance score for each path in the given list of paths.
    Args:
        path_list (list of list of tuples): A list where each element is a path, 
                                            and each path is a list of tuples representing points (x, y).
    Returns:
        list of float: A list of normalized average distance scores, one for each path.
    The function performs the following steps for each path:
    1. Removes all points that are mapped to (0, 0).
    2. Calculates the 1D distances (absolute difference between x and y) for each remaining point.
    3. Computes the average of these distances.
    4. Normalizes the average distance by dividing it by the maximum distance in the list.
    """
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