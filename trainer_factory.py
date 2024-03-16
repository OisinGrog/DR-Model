import os
import pytorch_lightning as pl
from pl_model import DR_model
from test_data import OisinDataset
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from collections import Counter
from utils import SaveMetricsCallback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

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

    def _fit(self, train_loader, valid_loader=None, test_loader=None, ckpt_path=None):
        self._log_hyperparameters()
        model_out_path = os.path.join(self.model_folder, f'{self.model_name}.ckpt')
        learner = DR_model(self.configfile_head['lr'])
        save_metrics_callback = SaveMetricsCallback(f"Resnet50_{self.configfile_head['lr']}_validation_metrics.json",
                                                    format='json')
        # Sample callback for Early Stopping
        """https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html"""
        training_callback = ModelCheckpoint(
            dirpath='callback-Model-outputs',
            verbose=True,
            monitor='valid/acc',
            mode='max'
        )

        early_stopping_callback = EarlyStopping(
            monitor="valid/acc",
            min_delta=0.01,
            verbose=True,
            patience=30,
            mode="max"
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
            callbacks=[save_metrics_callback, training_callback, early_stopping_callback]
        )
        if self.args.mode.lower() == 'test':
            fit_args = [learner, test_loader]
            return trainer.test(*fit_args, ckpt_path=ckpt_path)
        else:
            fit_args = [learner, train_loader, valid_loader] if self.configfile_head[
                                                                    'use_valid'].lower() == 'yes' else [
                learner,
                train_loader]
            if ckpt_path:
                if self.args.mode.lower() == 'train':
                    trainer.fit(*fit_args, ckpt_path=ckpt_path)
                    if trainer.interrupted:
                        self.logger.info('TERMINATED DUE TO INTERRUPTION')
                        trainer.save_checkpoint(model_out_path)
                        self.configfile.set('outputs', 'resume_ckpt', str(model_out_path))
                        self.configfile.set(f'outputs', 'output_model', str(model_out_path))

            else:
                trainer.fit(*fit_args)

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
        batch_size = int(self.configfile_head['batch_size'])  # Use the batch size from your config

        dr_dataset = OisinDataset(self.configfile_head['data_dir'], transform=transform)

        # Stratified split
        y = dr_dataset.df['diabetic_retinopathy'].values
        indices = range(len(dr_dataset))

        # Split dataset indices into train, validation, and test sets
        indices_train, indices_test, y_train, y_test = train_test_split(indices, y, test_size=0.1, stratify=y,
                                                                        random_state=42)
        indices_train, indices_val, y_train, y_val = train_test_split(indices_train, y_train, test_size=(2 / 9),
                                                                      stratify=y_train,
                                                                      random_state=42)  # Adjusting for the 20% split after removing 10% for test

        # Convert indices lists to arrays for easier manipulation
        indices_train_arr = np.array(indices_train)
        indices_val_arr = np.array(indices_val)
        indices_test_arr = np.array(indices_test)

        # Use the indices arrays to gather labels for each subset
        labels_train = y[indices_train_arr]
        labels_val = y[indices_val_arr]
        labels_test = y[indices_test_arr]

        # Count occurrences of each class in the subsets
        train_counts = np.bincount(labels_train)
        val_counts = np.bincount(labels_val)
        test_counts = np.bincount(labels_test)

        print(f"Training set: Class 0: {train_counts[0]}, Class 1: {train_counts[1]}")
        print(f"Validation set: Class 0: {val_counts[0]}, Class 1: {val_counts[1]}")
        print(f"Test set: Class 0: {test_counts[0]}, Class 1: {test_counts[1]}")

        # Creating Subset objects for train, validation, and test
        train_dataset = Subset(dr_dataset, indices_train)
        val_dataset = Subset(dr_dataset, indices_val)
        test_dataset = Subset(dr_dataset, indices_test)

        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")
        print(f"Test set size: {len(test_dataset)}")

        if self.args.mode.lower() == 'train':
            ckpt_path = None

            train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  num_workers=int(self.configfile_head['num_workers']))
            val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                num_workers=int(self.configfile_head['num_workers']))

            return self._fit(train_loader, valid_loader=val_loader, ckpt_path=ckpt_path)

        elif self.args.mode.lower() == 'test':
            print('Testing')
            ckpt_path = self.configfile['outputs']['output_model']
            test_loader = DataLoader(test_dataset, batch_size=batch_size,
                        num_workers=int(self.configfile_head['num_workers']))
            return self._fit(train_loader=None, test_loader=test_loader, ckpt_path=ckpt_path)

        elif self.args.mode.lower() == 'resume':
            ckpt_path = self.configfile['outputs']['resume_ckpt']
            train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  num_workers=int(self.configfile_head['num_workers']))
            val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                num_workers=int(self.configfile_head['num_workers']))
            return self._fit(train_loader, valid_loader=val_loader, ckpt_path=ckpt_path)

        else:
            raise Exception
