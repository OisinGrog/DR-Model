import os
from pathlib import Path
from trainer_factory import TrainerFactory
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import configparser
import argparse
import time
import logging
from utils import send_email, google_drive_service, file_upload, get_shareable_link


def get_logger(name):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M', level=logging.WARNING)
    logger = logging.getLogger(name)
    return logger


def train():
    logger = get_logger(__name__)
    parent_dir = (Path(os.getcwd()))
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', choices=['train', 'test', 'tune', 'resume'], default='train')
    args = parser.parse_args()
    service = google_drive_service()
    seed_everything(42, workers=True)
    tb_logger = TensorBoardLogger('training-logs', name='Diabetic-Retinopathy')
    configfile = configparser.ConfigParser()
    configfile.read('config.ini')

    configfile_head = configfile['hyperparameters']
    data_BS = int(configfile_head['batch_size'])
    learning_rate = float(configfile_head['lr'])
    epochs = int(configfile_head['epoch'])

    model_folder = os.path.join(parent_dir, 'OUTPUT')
    os.makedirs(model_folder, exist_ok=True) if not os.path.exists(model_folder) else None
    model_name = f'DR-epochs={epochs}-lr={learning_rate}-BS={data_BS}'
    trainer_factory = TrainerFactory(args, configfile, configfile_head, tb_logger, logger, model_name, model_folder,
                                     parent_dir)
    start_time = time.time()
    trainer_factory.get_train_trainer()
    end_time = time.time()
    total_time = end_time - start_time
    hours, minutes, seconds = total_time // 3600, (total_time % 3600) // 60, total_time % 60
    logger.info(f'Total Run time for Training is {int(hours)} hours, {int(minutes)}, and {seconds:.2f} seconds')

    model_path = os.path.join(model_folder, f'{model_name}.ckpt')

    model_id = file_upload(service, model_path, model_name)
    link = get_shareable_link(service, model_id)

    subject = "DR-Model Training Complete"
    body = f"Training has completed.\nTotal Run Time: {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds \n" \
           f"Please Download model here {link}"
    recipient_email = "samueladebayo@ieee.org"
    recipient_email_2 = 'sadebayo01@qub.ac.uk'
    recipient_email_3 = 'ogrogan02@qub.ac.uk'
    sender_email = "soluadebayo@gmail.com"
    sender_password = "*********"
    # Uncomment this part of the email to receive email update when training is done
    # send_email(recipient_email, sender_email, subject, body, password=sender_password)
    # logger.info(f'Email sent to {recipient_email}')
    # send_email(recipient_email_2, sender_email, subject, body, password=sender_password)
    # logger.info(f'Email sent to {recipient_email_2}')
    # send_email(recipient_email_3, sender_email, subject, body, password=sender_password)
    # logger.info(f'Email sent to {recipient_email_3}')


if __name__ == '__main__':
    train()
