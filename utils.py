# Imports
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
import time
from PIL import Image
import numpy as np
import argparse

# Function for data transformations
def data_transforms(data_dir):
    ''' Arguments: data main folder path
    
        Transforms and loads training, validation, and testing datasets
        
        Returns transformed datasets loaded with ImageFolder
    '''
    data_directory = data_dir
    train_dir = data_directory + '/train'
    valid_dir = data_directory + '/valid'
    test_dir = data_directory + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomVerticalFlip(),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                std=[0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                std=[0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    return train_data, validation_data, test_data

def data_loaders(data_dir):
    ''' Arguments: data main folder path
    
        Loads and transforms training, validation, and testing datasets
        
        Returns dataloaders
    '''
    data_directory = data_dir
    train_data, validation_data, test_data = data_transforms(data_directory)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return trainloader, validationloader, testloader

# Function for network setup
def network_setup(architecture='resnet101', dropout=0.2, input_units=2048, hidden_units=512, learning_rate=0.001, device='cuda'):
    ''' Arguments: network architecture (resnet101, densenet161 or vgg16), dropout rate,
        number of input and hidden units, learning rate, and device to use (cuda or cpu)
        
        Defines network architecture, hyperparameters, classifier, criterion and optimizer to
        use when training the network
        
        Returns model, criterion and optimizer
    '''
    if architecture == 'resnet101':
        model = models.resnet101(pretrained=True)
    elif architecture == 'densenet161':
        model = models.densenet161(pretrained=True)
    elif architecture == 'vgg16':
        model = models.vgg16(pretrained = True)
    else:
        print("{} is not a valid model. Please choose resnet101, densenet121 or vgg16.".format(architecture))
        
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(
        nn.Linear(input_units, hidden_units),
        nn.Dropout(dropout),
        nn.ReLU(),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1))
    
    model.fc = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), learning_rate)
    
    model.to(device)
    
    return model, criterion, optimizer

# Function for network training
def network_training(model, trainloader, validationloader, criterion, optimizer, epochs=18, print_every=15, device='cuda'):
    ''' Arguments: model, criterion, optimizer, number of epochs, number of steps before printing
        loss and accuracy (print_every), and device to use (cuda or cpu)
    
        Trains network and displays training and validation loss and validation accuracy
        
        Returns nothing
    '''
    steps = 0
    running_loss = 0
    
    start = time.time()
    print('Starting training')
    
    train_losses, validation_losses = [], []
    
    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1
            
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validationloader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)
                    
                        validation_loss += batch_loss.item()
                    
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                train_losses.append(running_loss/len(trainloader))
                validation_losses.append(validation_loss/len(validationloader))
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Training loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validationloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validationloader):.3f}")
            
                running_loss = 0
                model.train()
            
    time_elapsed = time.time() - start
    print('Time spent training: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

# Function for saving trained model
def save_checkpoint(model, train_data, optimizer, architecture='resnet101', dropout=0.2, input_units=2048, hidden_units=512, learning_rate=0.001, epochs=18, save_dir='checkpoint.pth'):
    ''' Arguments: architecture, dropout rate, number of input and hidden units, learning rate, number of epochs
        and checkpoint path name (save_dir)
    
        Saves trained model and hyperparameters necessary to rebuild model
        
        Returns nothing
    '''
    model.class_to_idx = train_data.class_to_idx
    model.cpu
    
    checkpoint = {'architecture': architecture,
                  'model': model,
                  'dropout': dropout,
                  'input_units': input_units,
                  'hidden_units': hidden_units,
                  'learning_rate': learning_rate,
                  'epochs': epochs,
                  'fc': model.fc,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()}

    torch.save(checkpoint, save_dir)

# Function for loading saved model
def load_checkpoint(save_dir='checkpoint.pth'):
    ''' Arguments: checkpoint path name (save_dir)
    
        Loads saved model
        
        Returns trained model with saved hyperparameters, weights and biases
    '''
    checkpoint = torch.load(save_dir)
    architecture = checkpoint['architecture']
    model = checkpoint['model']
    dropout = checkpoint['dropout']
    input_units = checkpoint['input_units']
    hidden_units = checkpoint['hidden_units']
    learning_rate = checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    model.fc = checkpoint['fc']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict = checkpoint['state_dict']
    model.optimizer = checkpoint['optimizer']
    
    return model

# Function for processing PIL image to use in model
def process_image(image):
    ''' Arguments: image path name
    
        Scales, crops, and normalizes a PIL image for a PyTorch model
        
        Returns a numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    
    image_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor()])
    
    pil_image = Image.open(image)
    pil_image = image_transforms(pil_image)
    
    np_image = np.array(pil_image)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return np_image

# Function for predicting probabilities and classes of image
def predict(image_path, model, topk=5, device='cuda'):
    ''' Arguments: image path name, model, number of predictions and device to use (cuda or cpu)
    
        Scales, crops, and normalizes a PIL image for a PyTorch model
        
        Returns probabilities and classes for topk most probable classes
    '''
    model.eval()
    model.to(device)
    
    image = Image.open(image_path)
    image = process_image(image_path)
    image = torch.FloatTensor(image)
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        logps = model.forward(image.to(device))
        ps = torch.exp(logps) # Using the same probability calculation as in model
        probs, classes = ps.topk(topk, dim=1)
        
        return probs, classes