"""
Tool Name: oct_ss_detection_learn
=================================

Description:
    Training tool for OCT Scleral Spur Detection using ResNet model. Trains a neural network 
    to predict scleral spur coordinates from OCT images.

Parameters:            
    - image_dir (str): Directory containing training images.
    - coord_file (str): CSV file with image names and corresponding coordinates.
    - output_model_path (str): Path where the trained model will be saved.
    - batch_size (int, optional): Training batch size. Default is 2.
    - learning_rate (float, optional): Learning rate for optimizer. Default is 0.005.
    - num_epochs (int, optional): Number of training epochs. Default is 100.
    - patience (int, optional): Early stopping patience. Default is 15.
    - val_split (float, optional): Validation split ratio. Default is 0.2.
    - box_size (int, optional): Size of bounding box for coordinate normalization. Default is 300.
    - device (str, optional): Device to train on ('cuda' or 'cpu'). Default is 'cuda' if available.
    - debug (bool, optional): Enable debug logging. Default is False.

Output:
    - str: Path to the saved trained model.
            
Example JSON Plan:
    "neural_net-oct_ss_detection_learn": {
        "code": "oct_ss_detection_learn.py",
        "image_dir": "/path/to/training/images",
        "coord_file": "/path/to/coordinates.csv",
        "output_model_path": "/path/to/save/best_model.pth",
        "batch_size": 4,
        "learning_rate": 0.005,
        "num_epochs": 100,
        "patience": 15,
        "debug": true
    }

Notes:
    - Trains a ResNet50 model for coordinate regression.
    - Uses MSE loss and Adam optimizer with learning rate scheduling.
    - Implements early stopping based on validation loss.
    - Automatically splits data into training and validation sets.
    - Saves the best model based on validation performance.
    - Coordinates in CSV should be normalized to the box_size.
"""

import asyncio
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
import csv
import logging

from sm_sample_processor import SMSampleProcessor
from sm_sample_id import SMSampleID


class CoordinateDataset(Dataset):
    def __init__(self, image_dir, coord_file, box_size=300, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.box_size = box_size
        self.data = []

        with open(coord_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header row
            for row in reader:   
                image_name, x, y = row
                self.data.append((image_name, float(x), float(y)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, x, y = self.data[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # normalize coordinates
        norm_x, norm_y = x / self.box_size, y / self.box_size

        return image, torch.tensor([norm_x, norm_y], dtype=torch.float32), image_name


class ResNetCoordinate(nn.Module):
    def __init__(self, num_outputs=2):
        super(ResNetCoordinate, self).__init__()
        self.resnet = models.resnet50(weights='DEFAULT')
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_outputs)  

    def forward(self, x):
        return self.resnet(x)


class OCTSSDetectionLearn(SMSampleProcessor):

    def __init__(self):
        super().__init__()

    def _setup_transforms(self):
        """Setup training and validation transforms."""
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform

    def _train_epoch(self, model, train_loader, criterion, optimizer, device, sample_id):
        """Train for one epoch."""
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, coordinates, _) in enumerate(train_loader):
            images, coordinates = images.to(device), coordinates.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, coordinates)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            if batch_idx % 10 == 0:  # Log every 10 batches
                self.print_log(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}", sample_id)

        return train_loss / len(train_loader)

    def _validate_epoch(self, model, val_loader, criterion, device):
        """Validate for one epoch."""
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, coordinates, _ in val_loader:
                images, coordinates = images.to(device), coordinates.to(device)
                outputs = model(images)
                loss = criterion(outputs, coordinates)
                val_loss += loss.item()

        return val_loss / len(val_loader)

    async def execute(
        self,
        *,
        image_dir: str,
        coord_file: str,
        output_model_path: str,
        batch_size: int = 2,
        learning_rate: float = 0.005,
        num_epochs: int = 100,
        patience: int = 15,
        val_split: float = 0.2,
        box_size: int = 300,
        device: str = "cuda",
        debug: bool = False,
        sample_id: SMSampleID
    ) -> str:

        if debug:
            self.print_log(f"Starting training with parameters:", sample_id)
            self.print_log(f"  Image directory: {image_dir}", sample_id)
            self.print_log(f"  Coordinate file: {coord_file}", sample_id)
            self.print_log(f"  Output model path: {output_model_path}", sample_id)
            self.print_log(f"  Batch size: {batch_size}", sample_id)
            self.print_log(f"  Learning rate: {learning_rate}", sample_id)
            self.print_log(f"  Number of epochs: {num_epochs}", sample_id)
            self.print_log(f"  Patience: {patience}", sample_id)

        try:
            # Setup device
            device_obj = torch.device(device if device == 'cpu' or torch.cuda.is_available() else 'cpu')
            self.print_log(f"Using device: {device_obj}", sample_id)

            # Setup transforms
            train_transform, val_transform = self._setup_transforms()

            # Create full dataset
            full_dataset = CoordinateDataset(image_dir, coord_file, box_size, transform=train_transform)
            dataset_size = len(full_dataset)
            val_size = int(val_split * dataset_size)
            train_size = dataset_size - val_size

            self.print_log(f"Dataset size: {dataset_size}, Train: {train_size}, Val: {val_size}", sample_id)

            # Split dataset
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
            
            # Update validation dataset transform
            val_dataset.dataset.transform = val_transform

            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Initialize model
            model = ResNetCoordinate().to(device_obj)
            
            # Setup training components
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

            # Training loop
            best_val_loss = float('inf')
            epochs_without_improvement = 0

            self.print_log("Starting training...", sample_id)

            for epoch in range(num_epochs):
                # Train
                avg_train_loss = self._train_epoch(model, train_loader, criterion, optimizer, device_obj, sample_id)
                
                # Validate
                avg_val_loss = self._validate_epoch(model, val_loader, criterion, device_obj)
                
                # Update scheduler
                scheduler.step(avg_val_loss)
                
                # Log progress
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else learning_rate
                self.print_log(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}', sample_id)

                # Early stopping and model saving
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_without_improvement = 0
                    
                    # Save best model
                    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'box_size': box_size,
                        'learning_rate': learning_rate
                    }, output_model_path)
                    
                    self.print_log(f"New best validation loss: {best_val_loss:.4f} - Model saved", sample_id)
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        self.print_log(f'Early stopping triggered after {epoch+1} epochs', sample_id)
                        break

            self.print_log(f"Training completed. Best validation loss: {best_val_loss:.4f}", sample_id)
            self.print_log(f"Model saved to: {output_model_path}", sample_id)
            
            return output_model_path

        except Exception as e:
            error_msg = f"Error during training: {str(e)}"
            self.print_error(error_msg, sample_id)
            raise RuntimeError(error_msg)


if __name__ == "__main__":
    tool = OCTSSDetectionLearn()
    asyncio.run(tool.main())