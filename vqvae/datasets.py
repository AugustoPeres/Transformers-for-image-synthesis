import os
import pandas as pd

import matplotlib.pyplot as plt

import torch
import torchvision


class ImageDataset(torch.utils.data.Dataset):
    """Dataset with images"""

    def __init__(self, path_to_csv):
        """
        Args:
          path_to_csv:
            path to a csv image descriptions and a relative path to the
            images.
        """
        super(ImageDataset).__init__()
        self.data = pd.read_csv(path_to_csv)
        self.length = len(self.data)

        self.csv_dir = os.path.dirname(path_to_csv)

    def transpose_image(self, image):
        """
        Torch read images in [image_channels, image_height, image_width].
        Here we change to [image_height, image_width, image_channels]
        """
        return image.transpose(-1, 0).transpose(0, 1)

    def plot_item(self, idx):
        plt.imshow(self.__getitem__(idx)['image'])
        plt.show()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = os.path.join(self.csv_dir, row['image_path'])
        image = torchvision.io.read_image(path)
        return {'image': self.transpose_image(image / 255)}
