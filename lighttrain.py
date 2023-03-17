import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader

# Create a dataset class
class MNISTDataModule(pl.LightningDataModule):
    # ... (same as before) ...
    pass
# Create a simple model class

class SimpleMNISTModel(pl.LightningModule):
    # ... (same as before) ...
    pass


def main():
    # Set up data module and model
    mnist_data = MNISTDataModule()
    mnist_model = SimpleMNISTModel()

    # Train the model using multiple GPUs
    num_gpus = torch.cuda.device_count()
    accelerator = "ddp" if num_gpus > 1 else None  # Use "ddp" (DistributedDataParallel) if multiple GPUs are available

    trainer = pl.Trainer(max_epochs=10, gpus=num_gpus, accelerator=accelerator)
    trainer.fit(mnist_model, mnist_data)

if __name__ == "__main__":
    main()