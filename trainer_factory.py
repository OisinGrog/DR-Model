import os
import pytorch_lightning as pl
from pl_model import DR_model
from test_data import OisinDataset
# from test_factory import Tester
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch

#shjshjs
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

    def _fit(self, train_loader, valid_loader=None, ckpt_path=None):
        self._log_hyperparameters()
        model_out_path = os.path.join(self.model_folder, f'{self.model_name}.ckpt')
        learner = DR_model(self.configfile_head['lr'])
        trainer = pl.Trainer(
            accelerator='cpu',
            devices=self.num_gpus,
            num_nodes=1,
            logger=self.tb_logger,
            sync_batchnorm=True,
            # deterministic=True,

            max_epochs=int(self.configfile_head['epoch'])
        )

        fit_args = [learner, train_loader, valid_loader] if self.configfile_head['use_valid'].lower() == 'yes' else [learner,
                                                                                                       train_loader]
        if ckpt_path:
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
        split = int(0.8 * total_samples)

        torch.manual_seed(42)  # Set a seed for reproducibility
        torch.cuda.manual_seed(42)  # If using GPU
        torch.cuda.manual_seed_all(42)  # If using multiple GPUs
        indices = torch.randperm(total_samples)

        # Split the indices into train and validation
        train_indices = indices[:split]
        val_indices = indices[split:]

        # Create SubsetRandomSampler for train and validation
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        # Create DataLoader for train and validation using the samplers
        batch_size = int(self.configfile_head['batch_size'])  # Use the batch size from your config
        train_loader = DataLoader(dr_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=int(self.configfile_head['num_workers']))
        val_loader = DataLoader(dr_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=int(self.configfile_head['num_workers']))
        if self.args.mode.lower() == 'train':
            ckpt_path = None
        elif self.args.mode.lower() == 'resume':
            ckpt_path = self.configfile['outputs']['resume_ckpt']

        # elif self.args.mode.lower() == 'test':
        #     tester = Tester(self.args, self.configfile_head)
        #     return tester.test(val_loader)
        else:
            raise Exception

        if val_loader is not None:
            return self._fit(train_loader, val_loader, ckpt_path=ckpt_path)
        else:
            return self._fit(train_loader, ckpt_path=ckpt_path)