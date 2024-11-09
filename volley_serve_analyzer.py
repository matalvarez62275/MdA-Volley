import time

from handle_files import *
from interface import *
from data import get_data

from metrics.metrics import metrics

valid = False
## Get user video
valid, video_path = get_video_name()

if valid:
    
    # Record the start time
    start_time = time.time()
    
    # Get user metrics
    metrics(video_path)
    
    # Get the data
    data = get_data(video_path)
    
    # Record the end time
    end_time = time.time()
    internal_time = end_time - start_time

    # Render detections
    render_detections(video_path)
    render_detections('PRO/videos/PRO1-right.mp4')
    
    # Plot results
    plot_time = graph_paths(data, video_path)
    
    # Calculate the elapsed time
    elapsed_time = internal_time + plot_time
    print(f"\nProcessing time: {elapsed_time:.2f} seconds\n")