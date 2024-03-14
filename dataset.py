from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import pandas as pd
import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms


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

        DR_label = row['diabetic_retinopathy']

        return {'image': image, 'DR_label': torch.tensor(DR_label, dtype=torch.long)}


if __name__ == '__main__':
    data_dir = '/home/qub-hri/Documents/Datasets/a-brazilian-multilabel-ophthalmological-dataset-brset-1.0.0/labels.csv'
    transform = transforms.Compose([transforms.ToTensor(),  # Add more transformations as needed
                                    ])

    dr_dataset = OisinDataset(data_dir, transform=transform)

    # Stratified split
    y = dr_dataset.df['diabetic_retinopathy'].values
    indices = range(len(dr_dataset))

    # Split dataset indices into train, validation, and test sets
    indices_train, indices_test, y_train, y_test = train_test_split(indices, y, test_size=0.1, stratify=y,
                                                                    random_state=42)
    indices_train, indices_val, y_train, y_val = train_test_split(indices_train, y_train, test_size=(2 / 9),
                                                                  stratify=y_train,
                                                                  random_state=42)  # Adjusting for the 20% split after removing 10% for test

    # Assuming indices_train, indices_val, indices_test are the indices for your train, validation, and test sets

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

    total_instances = 15220 + 1046
    class_0_weight = 1 / (15220 * total_instances)
    class_1_weight = 1 / (1046 * total_instances)

    # Normalize the weights such that the smallest weight is 1
    max_weight = max(class_0_weight, class_1_weight)
    class_0_weight = class_0_weight / max_weight
    class_1_weight = class_1_weight / max_weight

    class_weights = torch.tensor([class_0_weight, class_1_weight], dtype=torch.float32)
    print(f'Class Weight: {class_weights}')
