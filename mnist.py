from pathlib import Path
from typing import Optional, Union, Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir="./"):
        super().__init__()

        dataset = torchvision.datasets.MNIST(
            data_dir, train=True, download=True, transform=transforms.ToTensor()
        )
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, [55000, 5000]
        )
        self.test_dataset = torchvision.datasets.MNIST(
            data_dir, train=False, download=True, transform=transforms.ToTensor()
        )

        self.batch_size = batch_size
        self.data_dir = data_dir

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=8
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=8
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=8
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=8
        )
    
    
def main():
    print(1)
    print(100)
    
if __name__ == "__main_":
    main()