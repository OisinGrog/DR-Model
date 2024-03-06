import os
import pytorch_lightning as pl
from pl_model import DR_model
from test_data import OisinDataset
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from collections import Counter
import torch
from utils import SaveMetricsCallback
from pytorch_lightning.callbacks import ModelCheckpoint


class TrainerFactory:
    def __init__(self, args, configfile, configfile_head, tb_logger, logger, model_name, model_folder, parent_dir):
        self.args = args
        self.configfile = configfile
        self.configfile_head = configfile_head
        self.tb_logger = tb_logger
        self.logger = logger
        self.model_name = model_name
        self.model_folder = model_folder
        self.parent_dir = parent_dir

        self.epochs = int(self.configfile_head['epoch'])
        self.num_gpus = int(configfile_head['num_gpus'])

    def get_label(self, dataset_item):
        return dataset_item['DR_label'].item()

    def _fit(self, train_loader, valid_loader=None, ckpt_path=None):
        self._log_hyperparameters()
        model_out_path = os.path.join(self.model_folder, f'{self.model_name}.ckpt')
        learner = DR_model(self.configfile_head['lr'])
        save_metrics_callback = SaveMetricsCallback(f"Resnet50_{self.configfile_head['lr']}_validation_metrics.json",
                                                    format='json')
        # Sample callback for Early Stopping
        """https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html"""
        training_callback = ModelCheckpoint(
            dirpath='callback-Model-outputs',
            min_delta = '',
            patience = 10,
            verbose=True,
            mode = 'max'
        )

        # ModelCheckpoint(
        #     dirpath='callback-Model-outputs',
        #     filename='top100-{epoch:02d}-{valid/angular_error:.2f}',
        #     monitor='valid/acc',
        #     save_top_k=10,
        #     mode='max'
        # )


        trainer = pl.Trainer(
            accelerator='gpu',
            devices=self.num_gpus,
            num_nodes=1,
            logger=self.tb_logger,
            sync_batchnorm=True,
            max_epochs=int(self.configfile_head['epoch']),
            callbacks=[save_metrics_callback, training_callback]
        )
        if self.args.mode.lower() == 'test':
            fit_args = [learner, valid_loader]
            return trainer.test(*fit_args, ckpt_path=ckpt_path)
        else:
            fit_args = [learner, train_loader, valid_loader] if self.configfile_head[
                                                                    'use_valid'].lower() == 'yes' else [
                learner,
                train_loader]
            if ckpt_path:
                if self.args.mode.lower() == 'train':
                    trainer.fit(*fit_args, ckpt_path=ckpt_path)

            else:
                trainer.fit(*fit_args)

            if trainer.interrupted:
                self.logger.info('TERMINATED DUE TO INTERRUPTION')

            trainer.save_checkpoint(model_out_path)
            self.configfile.set('outputs', 'resume_ckpt', str(model_out_path))
            self.configfile.set(f'outputs', 'output_model', str(model_out_path))

            with open('config.ini', 'w') as f:
                self.configfile.write(f)

            self.logger.info(f'Full Checkpoint have been saved into {model_out_path}')
            self.logger.info(f'Training done!')

    def _log_hyperparameters(self):
        training_hyper = {'Batch Size': self.configfile_head['batch_size'], 'Learning Rate': self.configfile_head['lr'],
                          'Model Name': self.model_name,
                          'Model Output Folder': self.model_folder, 'EPOCH': self.epochs}
        [self.logger.info(f'{i}:{m}') for i, m in training_hyper.items()]

    def get_train_trainer(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        dr_dataset = OisinDataset(self.configfile_head['data_dir'], transform=transform)
        total_samples = len(dr_dataset)
        class_0_indices = dr_dataset.df[dr_dataset.df['diabetic_retinopathy'] == 0].index
        class_1_indices = dr_dataset.df[dr_dataset.df['diabetic_retinopathy'] == 1].index

        # Determine the number of samples per class for the validation set
        num_samples_per_class = min(len(class_0_indices), len(class_1_indices)) // 2  # Adjust as needed

        # Randomly select equal number of samples from each class
        np.random.seed(42)  # for reproducibility
        val_indices_class_0 = np.random.choice(class_0_indices, num_samples_per_class, replace=False)
        val_indices_class_1 = np.random.choice(class_1_indices, num_samples_per_class, replace=False)

        # Combine indices
        val_indices = np.concatenate((val_indices_class_0, val_indices_class_1))
        np.random.shuffle(val_indices)  # Shuffle the combined indices

        # Remaining indices for training
        train_indices = list(set(range(len(dr_dataset))) - set(val_indices))

        # Create Subset for training and validation
        train_dataset = Subset(dr_dataset, train_indices)
        val_dataset = Subset(dr_dataset, val_indices)
        val_class_counts = Counter(self.get_label(val_dataset[i]) for i in range(len(val_dataset)))

        print("Class Distribution in the Validation Dataset:")
        for class_label, count in val_class_counts.items():
            print(f'Class {class_label}: {count}')

        print("Class Distribution in the Validation Dataset:")
        print(f'Length of Train Dataset: {len(train_dataset)}')
        print(f'Length of Valid Dataset: {len(val_dataset)}')

        # Create DataLoader for train and validation using the samplers
        batch_size = int(self.configfile_head['batch_size'])  # Use the batch size from your config
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  num_workers=int(self.configfile_head['num_workers']))
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                num_workers=int(self.configfile_head['num_workers']))
        if self.args.mode.lower() == 'train':
            ckpt_path = None
        elif self.args.mode.lower() == 'test':
            ckpt_path = os.path.join(self.model_folder, f'{self.model_name}.ckpt') # self.configfile['outputs']['output_model']
        elif self.args.mode.lower() == 'resume':
            ckpt_path = self.configfile['outputs']['resume_ckpt']

        else:
            raise Exception

        if val_loader is not None:
            return self._fit(train_loader, val_loader, ckpt_path=ckpt_path)
        else:
            return self._fit(train_loader, ckpt_path=ckpt_path)
