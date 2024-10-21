import sys

from metrics.metrics import *

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from handle_files import get_video_name

valid = False

valid, video_path = get_video_name()
if valid:
    metrics(video_path)
    
