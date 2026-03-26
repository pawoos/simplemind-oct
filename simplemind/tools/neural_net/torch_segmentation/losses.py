import torch
from torch import nn
#from models import *

class MultiClassDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MultiClassDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # You may want to comment this out if your model contains a softmax or equivalent activation layer
        inputs = torch.nn.functional.softmax(inputs, dim=1)       

        # Create an empty tensor to store the Dice losses for each class
        dice_losses = torch.zeros(inputs.shape[1], device=inputs.device)

        for class_index in range(inputs.shape[1]):
            input_class = inputs[:, class_index]
            target_class = targets[:, class_index, :, :]
            intersection = (input_class * target_class).sum()
            dice_loss = 1 - (2. * intersection + smooth) / (input_class.sum() + target_class.sum() + smooth)
            dice_losses[class_index] = dice_loss

        return dice_losses.mean()



class MultiClassDiceCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MultiClassDiceCELoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets, smooth=1):
        # Dice Loss
        dice_loss = MultiClassDiceLoss()(inputs, targets, smooth)

        # Cross-Entropy Loss
        CE = self.ce(inputs, targets)

        # Combine the two loss functions
        Dice_CE = CE + dice_loss

        return Dice_CE


class DiceBCELoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.nn.functional.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        #print(f"targets type: {targets.dtype}, Shape: {targets.shape}, Device: {targets.device}", flush=True)
        BCE = torch.nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.nn.functional.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return dice_loss