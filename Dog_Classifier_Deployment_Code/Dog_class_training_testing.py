# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 2020
Author: Francisco Javier Carrera Arias
Dog Breed Tranfer Learning Classifier Training
"""
# Imports
import torch
import pickle as pkl
import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict
from torchvision import datasets, transforms
from utils import train_model, test_model
import torch.optim as optim

# Set PIL to be tolerant of image files that are truncated.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set to use CUDA
use_cuda = True

## Specify data loaders
data_dir = 'dogImages'

trainTrans = transforms.Compose([transforms.Resize(256),
                                       transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valTestTrans = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

train = datasets.ImageFolder(data_dir + '/train', transform=trainTrans)
val = datasets.ImageFolder(data_dir + '/valid', transform=valTestTrans)
test = datasets.ImageFolder(data_dir + '/test', transform=valTestTrans)

trainLoad = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
valLoad = torch.utils.data.DataLoader(val, batch_size=64, shuffle=True)
testLoad = torch.utils.data.DataLoader(test, batch_size=32)

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

if use_cuda:
    model_transfer = model_transfer.cuda()

# Set optimization criterion  
criterion_transfer = nn.NLLLoss()
optimizer_transfer = optim.RMSprop(model_transfer.fc.parameters(), lr=0.001)

loaders_transfer = {'train': trainLoad,'val': valLoad,'test': testLoad}
model_transfer = train_model(35, loaders_transfer, model_transfer, optimizer_transfer, 
                       criterion_transfer, use_cuda, 'model_transfer_best.pt')

# Test the model using the testing set
model_transfer.load_state_dict(torch.load('model_transfer_best.pt'))
test_model(loaders_transfer, model_transfer, criterion_transfer, use_cuda)

# Save the classes
class_names = [item[4:].replace("_", " ") for item in train.classes]
class_numbers = [item_element for item_element,item in enumerate(train.classes)]

# Save the class names and indices
with open('Dog_names.pkl', 'wb') as handle:
    pkl.dump(class_names, handle)

with open('Dog_indices.pkl', 'wb') as handle:
    pkl.dump(class_numbers, handle)