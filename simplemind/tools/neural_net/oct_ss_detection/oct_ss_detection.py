"""
Tool Name: oct_ss_detection
=================================

Description:
    OCT Scleral Spur Detection inference tool using a trained ResNet model. Predicts the coordinates 
    of the scleral spur in OCT images using a pre-trained neural network model.

Parameters:            
    - input_image (SMImage): Input OCT image for scleral spur detection.
    - model_path (str, optional): Path to the trained model file (.pth). Default uses weights/oct/scleral_spur.pth.
    - box_size (int, optional): Size of the bounding box used during training (for coordinate scaling). Default is 300. If input_image has crop_region metadata, this will be automatically derived from crop dimensions.
    - device (str, optional): Device to run inference on ('cuda' or 'cpu'). Default is 'cuda' if available.
    - debug (bool, optional): Enable debug logging. Default is False.

Output:
    - SMImage: The input image with scleral spur coordinates added to metadata.
            
Example JSON Plan:
    "neural_net-oct_ss_detection": {
        "code": "oct_ss_detection.py",
        "input_image": "from input_image",
        "device": "cuda",
        "debug": false
    }

Example JSON Plan (with explicit box_size):
    "neural_net-oct_ss_detection": {
        "code": "oct_ss_detection.py",
        "input_image": "from input_image",
        "box_size": 300,
        "device": "cuda",
        "debug": false
    }

Notes:
    - Uses pre-trained ResNet50 model for coordinate prediction (inference only).
    - Input image is automatically resized to 224x224 for model inference.
    - Predicted coordinates are scaled back to original image size using box_size.
    - If input image has crop_region metadata from a preceding crop tool, box_size is automatically derived from crop dimensions.
    - Coordinates are added to the output image metadata as 'scleral_spur_x' and 'scleral_spur_y'.
    - Model expects RGB input and applies standard ImageNet normalization.
    - Default model path: weights/oct/scleral_spur.pth
"""

import asyncio
import os
import torch 
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models

from sm_sample_processor import SMSampleProcessor
from sm_image import SMImage
from sm_sample_id import SMSampleID

class ResNetCoordinate(nn.Module):
    def __init__(self, num_outputs=2):
        super(ResNetCoordinate, self).__init__()
        self.resnet = models.resnet50(weights='DEFAULT')
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_outputs)  

    def forward(self, x):
        return self.resnet(x)


class OCTSSDetection(SMSampleProcessor):
    """OCT Scleral Spur Detection inference tool."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.device = None
        self.transform = None
        self.loaded_model_path = None

    def _get_default_model_path(self):
        """Get the default model path relative to the simplemind directory."""
        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate to simplemind root and then to weights/oct/scleral_spur.pth
        simplemind_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        default_path = os.path.join(simplemind_root, 'weights', 'oct', 'scleral_spur.pth')
        return default_path

    def _load_model(self, model_path: str, device: str):
        """Load the trained model from file."""
        # Only reload if different model path
        if self.model is not None and self.loaded_model_path == model_path:
            return
            
        self.device = torch.device(device if device == 'cpu' or torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = ResNetCoordinate().to(self.device)
        
        # Load trained weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        self.loaded_model_path = model_path

    def _setup_transform(self):
        """Setup image transformations for inference."""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _predict_coordinate(self, image_array: np.ndarray) -> tuple:
        """Predict scleral spur coordinates from image array."""
        # Convert numpy array to PIL Image
        if image_array.ndim == 2:
            # Grayscale to RGB
            pil_image = Image.fromarray(image_array).convert('RGB')
        elif image_array.ndim == 3:
            if image_array.shape[2] == 1:
                # Single channel to RGB
                pil_image = Image.fromarray(image_array[:, :, 0]).convert('RGB')
            else:
                # Already RGB/RGBA
                pil_image = Image.fromarray(image_array).convert('RGB')
        else:
            raise ValueError(f"Unsupported image dimensions: {image_array.shape}")

        # Apply transformations
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            pred_coords = output.squeeze().cpu().numpy()
        
        return float(pred_coords[0]), float(pred_coords[1])

    async def execute(
        self,
        *,
        input_image: SMImage,
        model_path: str = None,
        box_size: int = 300,
        device: str = "cuda",
        debug: bool = False,
        sample_id: SMSampleID
    ) -> SMImage:

        if input_image is None:
            return None

        # Use default model path if not provided
        if model_path is None:
            model_path = self._get_default_model_path()

        # Check for crop region metadata and derive box_size if available
        effective_box_size = box_size
        crop_width = None
        crop_height = None
        
        if input_image.metadata is not None and 'crop_region' in input_image.metadata:
            crop_region = input_image.metadata['crop_region']
            crop_width = crop_region.get('width')
            crop_height = crop_region.get('height')
            
            if crop_width is not None and crop_height is not None:
                # Use the maximum dimension as box_size (assuming square-like regions)
                # or use width if it's more relevant for scleral spur detection
                effective_box_size = max(crop_width, crop_height)
                if debug:
                    self.print_log(f"Found crop region metadata - width: {crop_width}, height: {crop_height}", sample_id)
                    self.print_log(f"Using derived box_size: {effective_box_size} (was {box_size})", sample_id)
            else:
                if debug:
                    self.print_log(f"Crop region metadata found but missing width/height, using default box_size: {box_size}", sample_id)
        else:
            if debug:
                self.print_log(f"No crop region metadata found, using provided box_size: {box_size}", sample_id)

        if debug:
            self.print_log(f"Input image shape: {input_image.pixel_array.shape}", sample_id)
            self.print_log(f"Model path: {model_path}", sample_id)
            self.print_log(f"Device: {device}", sample_id)
            self.print_log(f"Effective box_size: {effective_box_size}", sample_id)

        try:
            # Load model if not already loaded or different model
            if self.model is None or self.loaded_model_path != model_path:
                self._load_model(model_path, device)
                self._setup_transform()
                if debug:
                    self.print_log(f"Model loaded successfully on {self.device}", sample_id)

            # Get input image array
            input_array = input_image.pixel_array
            
            # Normalize shape - squeeze out singleton dimensions
            input_array = np.squeeze(input_array)
            
            # Ensure proper data type
            if input_array.dtype != np.uint8:
                # Normalize to 0-255 range if needed
                if input_array.max() <= 1.0:
                    input_array = (input_array * 255).astype(np.uint8)
                else:
                    input_array = input_array.astype(np.uint8)

            if debug:
                self.print_log(f"Processed image shape: {input_array.shape}, dtype: {input_array.dtype}", sample_id)

            # Predict scleral spur coordinates
            norm_x, norm_y = self._predict_coordinate(input_array)
            
            # Scale coordinates back to original size using effective box_size
            pred_x = norm_x * effective_box_size
            pred_y = norm_y * effective_box_size

            if debug:
                self.print_log(f"Normalized coordinates: ({norm_x:.4f}, {norm_y:.4f})", sample_id)
                self.print_log(f"Scaled coordinates: ({pred_x:.2f}, {pred_y:.2f}) using box_size: {effective_box_size}", sample_id)

            # Prepare metadata
            new_metadata = None
            if input_image.metadata is not None:
                new_metadata = input_image.metadata.copy()
            else:
                new_metadata = {}
            
            # Add scleral spur coordinates to metadata
            new_metadata.update({
                "scleral_spur_x": pred_x,
                "scleral_spur_y": pred_y,
                "scleral_spur_detected": True,
                "box_size": effective_box_size,
                "box_size_source": "crop_region" if (crop_width is not None and crop_height is not None) else "user_input"
            })

            if crop_width is not None and crop_height is not None:
                new_metadata.update({
                    "crop_width_used": crop_width,
                    "crop_height_used": crop_height
                })

            self.print_log(f"Scleral spur detection complete - coordinates: ({pred_x:.2f}, {pred_y:.2f}), box_size: {effective_box_size}", sample_id)
            
            # Return the same image with updated metadata
            return SMImage(new_metadata, input_image.pixel_array, input_image.label_array)

        except Exception as e:
            error_msg = f"Error in scleral spur detection: {str(e)}"
            self.print_error(error_msg, sample_id)
            raise RuntimeError(error_msg)


if __name__ == "__main__":
    tool = OCTSSDetection()
    asyncio.run(tool.main())