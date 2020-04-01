# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 23:12:22 2020

@author: alpha
"""

import pickle
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict
from PIL import Image
from PIL import ImageFile

def predict_breed_transfer(image_path, best_model = 'model_transfer_best.pt',
                           Dog_names = "Dog_names.pkl",Dog_indices = "Dog_indices.pkl"):
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    # class names without number (Affenpinscher, Brussels griffon...)
    # class names with number (001.Affenpinscher', '038.Brussels_griffon')
    with open(Dog_names, 'rb') as handle:
        class_names = pickle.load(handle)
    with open(Dog_indices, 'rb') as handle:
        class_numbers = pickle.load(handle)
    
    # Load and normalize Image
    Im = Image.open(image_path)
    trans = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    Image_Norm = trans(Im)
    Image_Norm = Image_Norm.unsqueeze_(0)
    
    # Instantiate the model
    # Specify model architecture 
    # Import the pretrained version of ResNet 50
    model_transfer = models.resnet50(pretrained=True)

    for param in model_transfer.parameters():
        param.requires_grad = False
    
    # Modify the last fully connected layar to make it relevant to the new training dataset
    classifier = nn.Sequential(OrderedDict([
                          ('h1',nn.Linear(2048,1024)),
                          ('relu1', nn.ReLU()), 
                          ('drop1',nn.Dropout(0.2)),
                          ('h2', nn.Linear(1024, 512)),
                          ('relu2', nn.ReLU()),
                          ('drop2',nn.Dropout(0.2)),
                          ('h3', nn.Linear(512, 133)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model_transfer.fc = classifier
    
    # Load best model
    model_transfer.load_state_dict(torch.load(best_model, map_location = "cpu"))
    
    # Switch to eval model
    model_transfer.eval()
    
    # Get output
    output = model_transfer(Image_Norm)
    
    # Get three breeds which has the highest probabilities.
    ps = torch.exp(output)
    top_p, top_class = ps.topk(3,dim = 1)
    
    # Get the names of breed for displaying
    Dogs = [class_names[class_numbers[i]] for i in top_class[0]]
    
    # Get the probabilities as a form of Tensor
    probs = top_p[0]
    probs = probs.data.numpy()
    
    return Dogs, probs