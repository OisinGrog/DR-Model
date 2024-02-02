import pickle
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
import smtplib
from email.mime.text import MIMEText
from tqdm import tqdm
import io
import configparser
from pytorch_lightning.callbacks import Callback
import json
import csv


def modify_config(learning_rate):
    config = configparser.ConfigParser()
    config.read('config.ini')
    config['hyperparameters']['lr'] = str(learning_rate)
    with open('config.ini', 'w') as configfile:
        config.write(configfile)


def google_drive_service():
    """
    Authenticate and create a service for Google Drive.
    :return:
    """

    SCOPES = ['https://www.googleapis.com/auth/drive']
    creds = None

    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    # If no file token.pickle is found request user's access and refresh tokens

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the new credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return build('drive', 'v3', credentials=creds)


def file_upload(service, file_path, file_name):
    """Upload a file to Google Drive with a progress bar."""
    file_metadata = {'name': file_name}
    file_size = os.path.getsize(file_path)

    with open(file_path, 'rb') as file:
        media = MediaIoBaseUpload(io.BytesIO(file.read()), mimetype='application/octet-stream', chunksize=1024 * 1024,
                                  resumable=True)
        request = service.files().create(body=file_metadata, media_body=media, fields='id')

        response = None
        progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True)

        while response is None:
            status, response = request.next_chunk()
            if status:
                progress_bar.update(status.resumable_progress - progress_bar.n)

        progress_bar.close()

    return response.get('id')


def get_shareable_link(service, file_id):
    """Create a shareable link for the file."""
    service.permissions().create(
        fileId=file_id,
        body={"role": "reader", "type": "anyone"}
    ).execute()
    return f"https://drive.google.com/uc?id={file_id}"


def send_email(recipient, smtp_user, subject, body, smtp_port=587, password='password', smtp_server='smtp.gmail.com'):
    """Send an email with the link."""
    # Set up your SMTP server and credentials

    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(smtp_user, password)

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = smtp_user
    msg['To'] = recipient

    server.sendmail(smtp_user, recipient, msg.as_string())
    server.quit()


class SaveMetricsCallback(Callback):
    def __init__(self, filename, format='json'):
        self.filename = filename
        self.format = format
        self.metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        accuracy = trainer.callback_metrics.get('valid/accuracy').item()
        epoch = trainer.current_epoch
        if any(metric['epoch'] == epoch for metric in self.metrics):
            for metric in self.metrics:
                if metric['epoch'] == epoch:
                    metric['validation_accuracy'] = accuracy
                    break
        else:
            self.metrics.append({'epoch': epoch, 'validation_accuracy': accuracy})

        # metric = {'epoch': epoch, 'validation_accuracy': accuracy}
        # self.metrics.append(metric)

        if self.format == 'json':
            with open(self.filename, 'w') as f:
                json.dump(self.metrics, f, indent=4)

        elif self.format == 'csv':
            with open(self.filename, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['epoch', 'validation_accuracy'])
                if not os.path.exists(self.filename) or os.stat(self.filename).st_size == 0:
                    writer.writeheader()
                writer.writerow({'epoch': epoch, 'validation_accuracy': accuracy})

