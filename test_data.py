import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
from torchvision import transforms
from PIL import Image
from pathlib import Path


class OisinDataset(Dataset):
    def __init__(self, csv_directory: Path, transform=None):
        """

        :param directory:
        :param transform:
        :param num_features:
        """
        self.data_path = Path(csv_directory)
        self.image_parent = os.path.join(self.data_path.parent, 'fundus_photos')
        self.transform = transform
        self.lines = []

        with open(self.data_path, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                self.lines.append(line)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].split(',')
        image_path = line[0]
        label = np.array(line[20]).astype('float')
        image_path = os.path.join(self.image_parent, f'{image_path}.jpg')
        retina_image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            retina_image = self.transform(retina_image)
        DR_label = torch.from_numpy(label).type(torch.FloatTensor)
        data_sample = {'image': retina_image, 'DR_label': DR_label}
        return data_sample


# if __name__ == '__main__':
#     dataset_directory = r'C:\Users\oisin\Documents\University\3rd Year\Final Year Project\Retinal Imaging FYP Shared Folder\Datasets\Brazilian\full_data_set\labels.csv'
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor()
#     ])
#     ucf_dataset = OisinDataset(dataset_directory, transform=transform,)
#     batch_size = 4
#     ucf_dataloader = DataLoader(ucf_dataset, batch_size=batch_size, shuffle=True)
#     for i, batch in enumerate(ucf_dataloader):
#         frames = batch['image']
#         labels = batch['DR_label']
#
#         print(f'Batch {i}:')
#         print(f'Frames Shape: {frames.shape}')
#         print(f'Labels:{labels}')
