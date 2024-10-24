import pandas as pd

def get_player_data(csv_name):
    """
    Reads player data from a CSV file and returns it as a list of lists.
    Args:
        csv_name (str): The path to the CSV file containing player data.
    Returns:
        list: A list containing three lists:
            - time (list): A list of time indices corresponding to each row in the CSV.
            - arm (list): A list of arm angles extracted from the 'Arm Angle' column.
            - fingers (list): A list of finger distances extracted from the 'Finger Distance' column.
    """
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_name)

    # Access each column and convert it to a Python list
    arm = df['Arm Angle'].tolist()
    fingers = df['Finger Distance'].tolist()
    time = [i for i in range(len(arm))]
    data = [time, arm, fingers]
    return data

def get_dataframe(pro_data, user_data):
    """
    Create a pandas DataFrame from professional and user data.
    Parameters:
    pro_data (list of lists): A list containing three lists for professional data:
        - pro_data[0]: List of time values for professional data.
        - pro_data[1]: List of arm angle values for professional data.
        - pro_data[2]: List of finger distance values for professional data.
    user_data (list of lists): A list containing three lists for user data:
        - user_data[0]: List of time values for user data.
        - user_data[1]: List of arm angle values for user data.
        - user_data[2]: List of finger distance values for user data.
    Returns:
    pd.DataFrame: A DataFrame with columns 'Player', 'Time', 'Arm Angle', and 'Finger Distance'.
        The 'Player' column indicates whether the data is from a professional or a user.
    """

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
    
    return pd.DataFrame(data)

def get_data(video_path):
    """
    Retrieves and processes player data from specified video path.

    This function extracts player data for both professional and user players,
    processes the data, and returns it in a dataframe format.

    Args:
        video_path (str): The file path to the user's video file.

    Returns:
        pandas.DataFrame: A dataframe containing the processed player data.
    """
    pro_data = get_player_data('PRO/metrics/PRO1-right.csv')
    user_data = get_player_data('USER/metrics/' + video_path.split('/')[-1][:-4] + '.csv')
    data = get_dataframe(pro_data, user_data)
    return data
