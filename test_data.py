import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from pathlib import Path


class OisinDataset(Dataset):
    def __init__(self, csv_directory: Path, transform=None):
        """
        :param csv_directory: Path to the directory containing the CSV file.
        :param transform: Transformations to be applied to the images.
        """
        self.data_path = Path(csv_directory)
        self.image_parent = os.path.join(self.data_path.parent, 'fundus_photos')
        self.transform = transform

        # Read the CSV file into a DataFrame
        self.df = pd.read_csv(csv_directory)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        row = self.df.iloc[idx]
        # rest of your code

        image_path = os.path.join(self.image_parent, f"{row['image_id']}.jpg")

        # Handle missing images
        if not os.path.exists(image_path):
            print(f'{image_path} does not exist')
            return None

        # Load the image
        image = Image.open(image_path).convert('RGB')

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Replace 'diabetic_retinopathy' with the correct column name for DR label
        DR_label = row['diabetic_retinopathy']

        return {'image': image, 'DR_label': torch.tensor(DR_label, dtype=torch.long)}

# Example usage
# transform = transforms.Compose([...])  # Define your transformations here
# dataset = OisinDataset(csv_directory='path/to/csv', transform=transform)
# loader = DataLoader(dataset, batch_size=32, shuffle=True)
