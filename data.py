import pandas as pd

def get_player_data(csv_name):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_name)

    # Access each column and convert it to a Python list
    arm = df['Arm Angle'].tolist()
    fingers = df['Finger Distance'].tolist()
    time = [i for i in range(len(arm))]
    data = [time, arm, fingers]
    return data

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
    
    return pd.DataFrame(data)

def get_data(video_path):
    pro_data = get_player_data('PRO/metrics/PRO1-right.csv')
    user_data = get_player_data('USER/metrics/' + video_path.split('/')[-1][:-4] + '.csv')
    data = get_dataframe(pro_data, user_data)
    return data