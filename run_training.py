import subprocess
from utils import send_email, google_drive_service, file_upload, get_shareable_link, modify_config
import sys
import subprocess

if __name__ == "__main__":
    learning_rate = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "train"
    modify_config(learning_rate)
    subprocess.run(['python', 'trainer.py', '--mode', mode])
