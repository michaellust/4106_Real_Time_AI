# -*- coding: utf-8 -*-
"""ECGR4_4106_Homework#3_MichaelLust.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HV0wF0RrAyya2BLtShORFWfqgZTPFCgJ
"""

#ECGR 4106: Real-Time AI
#Michael Lust
#Homework 3
#March 29, 2022

#Convolution online example:
#https://shonit2096.medium.com/cnn-on-cifar10-data-set-using-pytorch-34be87e09844
#Used to help construct CNN to clasify images across all 10 classes in CIFER10

"""a. Build a Convolutional Neural Network, like what we built in lectures (without skip connections), to classify the images across all 10 classes in CIFAR 10. You need to adjust the fully connected layer at the end properly with respect to the number of output classes. Train your network for 300 epochs. Report your training time, training loss, and evaluation accuracy after 300 epochs. Analyze your results in your report and compare them against a fully connected network (homework 2) on training time, achieved accuracy, and model size. Make sure to submit your code by providing the GitHub URL of your course repository for this course."""

# Commented out IPython magic to ensure Python compatibility.
#Imports
# %matplotlib inline
from matplotlib import pyplot as plt
import numpy as np
import collections

#Pytorch imports
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

device = torch.device('cuda:0')
print(torch.cuda.is_available())

#Using google colab assigned GPU to train
#device = !nvidia-smi
#device = '\n'.join(device)
#if device.find('failed') >= 0:
  #print('Not connected to a GPU')
#else:
  #print(device)

# number of subprocesses to use for data loading
num_workers = 0

# how many samples per batch to load
batch_size = 20

# percentage of training set to use as validation
valid_size = 0.2

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# choose the training and test datasets
train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)

# specify the image classes
classes = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))

# display 20 images
for idx in np.arange(10):
  ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])
  imshow(images[idx])
  ax.set_title(classes[labels[idx]])

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# define the CNN architecture
#Building class similar to: model = nn.Sequential from textbook example

#Online example
class Net1(nn.Module):
  def __init__(self):
    super(Net1, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

#Textbook example
# class Net1(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
#         self.act1 = nn.Tanh()
#         self.pool1 = nn.MaxPool2d(2)
#         self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
#         self.act2 = nn.Tanh()
#         self.pool2 = nn.MaxPool2d(2)
#         self.fc1 = nn.Linear(8 * 8 * 8, 32) #The size of the first linear layer is dependent on the expected size of the output of MaxPool2d: 8 × 8 × 8 = 512. 
#         self.act3 = nn.Tanh()
#         self.fc2 = nn.Linear(32, 2)

    # def forward(self, x):
    #     out = self.pool1(self.act1(self.conv1(x)))
    #     out = self.pool2(self.act2(self.conv2(out)))
    #     out = out.view(-1, 8 * 8 * 8) # <1> Forwarding the inputs of the model and reshaping it to return the output
    #     out = self.act3(self.fc1(out))
    #     out = self.fc2(out)
    #     return out

# create a complete CNN
model1 = Net1()
if torch.cuda.is_available():
    model1.cuda()
print(model1)

numel_list = [p.numel() for p in model1.parameters()]
sum(numel_list), numel_list

import torch.optim as optim

# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.SGD(model1.parameters(), lr=.01)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# from tqdm import tqdm
# #to record model runtime
# # number of epochs to train the model
# n_epochs = 300 #Needs to be 300
# 
# #List to store loss to visualize
# train_losslist = []
# 
# # track change in validation loss
# valid_loss_min = np.Inf
# 
# for epoch in range(1, n_epochs+1):
#   if epoch == 1 or epoch % 30 == 0: 
#     print('\nEpoch: {}'.format(epoch))
# 
#   # keep track of training and validation loss
#   train_loss = 0.0
#   valid_loss = 0.0
# 
#   if epoch == 1 or epoch % 30 == 0: 
#     print('training: ')  
# 
#   with tqdm(train_loader, unit="batch") as tepoch:  
#     # train the model
#     model1.train()
#     for data, target in tepoch:
# 
#         # move tensors to GPU if CUDA is available
#         data, target = data.to(device), target.to(device)
# 
#         # clear the gradients of all optimized variables
#         optimizer.zero_grad()
#         
#         # forward pass: compute predicted outputs by passing inputs to the model
#         output = model1(data)
#         
#         # calculate the batch loss
#         loss = criterion(output, target)
#         
#         # backward pass: compute gradient of the loss with respect to model parameters
#         loss.backward()
#         
#         # perform a single optimization step (parameter update)
#         optimizer.step()
#         
#         # update training loss
#         train_loss += loss.item()*data.size(0)
# 
#   if epoch == 1 or epoch % 30 == 0:       
#    print('validation: ')      
# 
#   with tqdm(valid_loader, unit="batch") as vepoch:
#     # validate the model
#     model1.eval()
#     for data, target in vepoch:
# 
#         # move tensors to GPU if CUDA is available
#         data, target = data.to(device), target.to(device)
#         
#         # forward pass: compute predicted outputs by passing inputs to the model
#         output = model1(data)
#         
#         # calculate the batch loss
#         loss = criterion(output, target)
#         
#         # update average validation loss 
#         valid_loss += loss.item()*data.size(0)
#     
#   # calculate average losses
#   train_loss = train_loss/len(train_loader.dataset)
#   valid_loss = valid_loss/len(valid_loader.dataset)
#   train_losslist.append(train_loss)
#         
#   # print training/validation statistics 
#   if epoch == 1 or epoch % 30 == 0: 
#     print('Training Loss: {:.6f} \tValidation Loss: {:.6f}'.format(train_loss, valid_loss))
#   
#   # save model if validation loss has decreased
#   if epoch == 1 or epoch % 30 == 0: 
#     if valid_loss <= valid_loss_min:
#       print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
#       valid_loss_min,
#       valid_loss))
#       torch.save(model1.state_dict(), 'model_cifar.pt')
#       valid_loss_min = valid_loss
# 
# # plt.plot(n_epochs, train_losslist)
# # plt.xlabel("Epoch")
# # plt.ylabel("Loss")
# # plt.title("Performance of Model 1")
# # plt.show()

model1.load_state_dict(torch.load('model_cifar.pt'))

# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model1.eval()
# iterate over test data
for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    data, target = data.to(device), target.to(device)
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model1(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not device else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

"""#### b. Extend your CNN by adding one more additional convolution layer followed by an activation function and pooling function. You also need to adjust your fully connected layer properly with respect to intermediate feature dimensions. Train your network for 300 epochs. Report your training time, loss, and evaluation accuracy after 300 epochs. Analyze your results in your report and compare your model size and accuracy over the baseline implementation in Problem1.a. Do you see any over-fitting? Make sure to submit your code by providing the GitHub URL of your course repository for this course."""

import torch.nn as nn
import torch.nn.functional as F
# define the CNN architecture
#Building class similar to: model = nn.Sequential from textbook example but for convolution

#Online example
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1) #Adding one more layer to compare to Net1
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)
        # dropout
        self.dropout = nn.Dropout(p=.5)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x))) #Adding one more layer to compare to Net1
        # flattening
        x = x.view(-1, 64 * 4 * 4)
        # fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# create a complete CNN
model2 = Net2()
if torch.cuda.is_available():
    model2.cuda()
print(model2)

numel_list = [p.numel() for p in model2.parameters()]
sum(numel_list), numel_list

import torch.optim as optim

# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.SGD(model2.parameters(), lr=.01)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# from tqdm import tqdm
# 
# # number of epochs to train the model
# n_epochs = 300 #Needs to be 300
# 
# #List to store loss to visualize
# train_losslist = []
# 
# # track change in validation loss
# valid_loss_min = np.Inf
# 
# for epoch in range(1, n_epochs+1):
#   print('\nEpoch: {}'.format(epoch))
# 
#   # keep track of training and validation loss
#   train_loss = 0.0
#   valid_loss = 0.0
# 
#   if epoch == 1 or epoch % 30 == 0: 
#     print('training: ') 
# 
#   with tqdm(train_loader, unit="batch") as tepoch:  
#     # train the model
#     model2.train()
#     for data, target in tepoch:
# 
#         # move tensors to GPU if CUDA is available
#         data, target = data.to(device), target.to(device)
# 
#         # clear the gradients of all optimized variables
#         optimizer.zero_grad()
#         
#         # forward pass: compute predicted outputs by passing inputs to the model
#         output = model2(data)
#         
#         # calculate the batch loss
#         loss = criterion(output, target)
#         
#         # backward pass: compute gradient of the loss with respect to model parameters
#         loss.backward()
#         
#         # perform a single optimization step (parameter update)
#         optimizer.step()
#         
#         # update training loss
#         train_loss += loss.item()*data.size(0)
# 
#   if epoch == 1 or epoch % 30 == 0:      
#     print('validation: ')     
# 
#   with tqdm(valid_loader, unit="batch") as vepoch:
#     # validate the model
#     model2.eval()
#     for data, target in vepoch:
# 
#         # move tensors to GPU if CUDA is available
#         data, target = data.to(device), target.to(device)
#         
#         # forward pass: compute predicted outputs by passing inputs to the model
#         output = model2(data)
#         
#         # calculate the batch loss
#         loss = criterion(output, target)
#         
#         # update average validation loss 
#         valid_loss += loss.item()*data.size(0)
#     
#   # calculate average losses
#   train_loss = train_loss/len(train_loader.dataset)
#   valid_loss = valid_loss/len(valid_loader.dataset)
#   train_losslist.append(train_loss)
#         
#   # print training/validation statistics
#   if epoch == 1 or epoch % 30 == 0: 
#     print('Training Loss: {:.6f} \tValidation Loss: {:.6f}'.format(train_loss, valid_loss))
# 
#   # save model if validation loss has decreased
#   if epoch == 1 or epoch % 30 == 0: 
#     if valid_loss <= valid_loss_min:
#       print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
#       valid_loss_min,
#       valid_loss))
#       torch.save(model2.state_dict(), 'model_cifar.pt')
#       valid_loss_min = valid_loss
# 
# # plt.plot(n_epochs, train_losslist)
# # plt.xlabel("Epoch")
# # plt.ylabel("Loss")
# # plt.title("Performance of Model 1")
# # plt.show()

model2.load_state_dict(torch.load('model_cifar.pt'))

# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model2.eval()
# iterate over test data
for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    data, target = data.to(device), target.to(device)
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model2(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not device else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

"""a. Build a ResNet based Convolutional Neural Network, like what we built in lectures (with skip connections), to classify the images across all 10 classes in CIFAR 10. For this problem, let’s use 10 blocks for ResNet and call it ResNet-10. Use the similar dimensions and channels as we need in lectures. Train your network for 300 epochs. Report your training time, training loss, and evaluation accuracy after 300 epochs. Analyze your results in your report and compare them against problem 1.b on training time, achieved accuracy, and model size. Make sure to submit your code by providing the GitHub URL of your course repository for this course."""

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

from torchvision import datasets, transforms
data_path = '../data-unversioned/p1ch6/'

#Training Set
cifar10_train = datasets.CIFAR10(
    data_path, train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))

#Validation Set
cifar10_val = datasets.CIFAR10(
    data_path, train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))

#Loading the Data properly to follow textbook example
train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=64,
                                           shuffle=False)
valid_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=64,
                                         shuffle=False)
all_acc_dict = collections.OrderedDict()

#Creating training loop function to run example from lecture
import datetime

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 25 == 0:
            print('\nEpoch {}, \nTraining loss {}'.format(epoch,
                loss_train / len(train_loader)))
            validate(model, train_loader, valid_loader)

#Defining Validation training model
def validate(model, train_loader, val_loader):
    accdict = {}
    for name, loader in [("train", train_loader), ("val", valid_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in tqdm(loader):
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1) # <1>
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        print("Accuracy {}: {:.2f}".format(name , correct / total))
        accdict[name] = correct / total
    return accdict

#Using the texbook example that is also found in the lecture slides
#NetRes is replaced by NetResDeep to be used for the model.
class NetRes(nn.Module):
    def __init__(self, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(n_chans1 // 2, n_chans1 // 2,
                               kernel_size=3, padding=1)
        self.fc1 = nn.Linear(4 * 4 * n_chans1 // 2, 32)
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = F.max_pool2d(torch.relu(self.conv2(out)), 2)
        out1 = out
        out = F.max_pool2d(torch.relu(self.conv3(out)) + out1, 2)
        out = out.view(-1, 4 * 4 * self.n_chans1 // 2)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

#Using the texbook example that is also found in the lecture slides

class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3,
                              padding=1, bias=False)  # <1>
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)
        torch.nn.init.kaiming_normal_(self.conv.weight,
                                      nonlinearity='relu')  # <2>
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x

#Using the texbook example that is also found in the lecture slides
#Constructing ResNet to handle 10 blocks

#Online example used as reference to constructing ResNet and ResNetBlock
# class Net2(nn.Module):
#     def __init__(self):
#         super(Net2, self).__init__()
#         # convolutional layer
#         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, 3, padding=1) #Adding one more layer to compare to Net1
#         # max pooling layer
#         self.pool = nn.MaxPool2d(2, 2)
#         # fully connected layers
#         self.fc1 = nn.Linear(64 * 4 * 4, 512)
#         self.fc2 = nn.Linear(512, 64)
#         self.fc3 = nn.Linear(64, 10)
#         # dropout
#         self.dropout = nn.Dropout(p=.5)

#     def forward(self, x):
#         # add sequence of convolutional and max pooling layers
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x))) #Adding one more layer to compare to Net1
#         # flattening
#         x = x.view(-1, 64 * 4 * 4)
#         # fully connected layers
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.dropout(F.relu(self.fc2(x)))
#         x = self.fc3(x)
#         return x

class NetResDeep(nn.Module):
    def __init__(self, n_chans1=32, n_blocks=10):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.resblocks = nn.Sequential(
            *(n_blocks * [ResBlock(n_chans=n_chans1)]))
        self.fc1 = nn.Linear(8 * 8 * n_chans1, 32)
        self.fc2 = nn.Linear(32, 10)
        
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 8 * 8 * self.n_chans1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# create a complete CNN
ResNet_10 = NetResDeep(n_chans1=32, n_blocks=10).to(device=device)
optimizer = optim.SGD(ResNet_10.parameters(), lr=3e-3)
criterion = nn.CrossEntropyLoss() # loss function

# Commented out IPython magic to ensure Python compatibility.
# #Doing the training loop
# %%time
# training_loop(
#     n_epochs = 300,
#     optimizer = optimizer,
#     model = ResNet_10,
#     loss_fn = criterion,
#     train_loader = train_loader,
# )
# all_acc_dict["res deep"] = validate(ResNet_10, train_loader, val_loader)

"""b. Develop three additional trainings and evaluations for your ResNet-10 to assess the impacts of regularization to your ResNet-10.

Weight Decay with lambda of 0.001
Dropout with p=0.3
Batch Normalization
Report and compare your training time, training loss, and evaluation accuracy after 300 epochs across these three different trainings.
"""

#Using weight penalization with a lambda of 0.001

def training_loop_l2reg(n_epochs, optimizer, model, loss_fn,
                        train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum()
                          for p in model.parameters())  # <1>
            loss = loss + l2_lambda * l2_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item()

        if epoch == 1 or epoch % 30 == 0:
            print('\nEpoch {}, \nTraining loss {}'.format(epoch,
                loss_train / len(train_loader)))
            validate(model, train_loader, valid_loader)

#Complete CNN with weight decay
ResNet_10 = NetResDeep(n_chans1=32, n_blocks=10).to(device=device)
#ResNet_10 = Net1().to(device=device)
optimizer = optim.SGD(ResNet_10.parameters(), lr=3e-3)
criterion = nn.CrossEntropyLoss() # loss function

# Commented out IPython magic to ensure Python compatibility.
# %%time
# #Doing the training loop with weight decay
# training_loop_l2reg(
#     n_epochs = 300, #Needs to be 300
#     optimizer = optimizer,
#     model = ResNet_10,
#     loss_fn = criterion,
#     train_loader = train_loader,
# )
# all_acc_dict["weight_decay"] = validate(ResNet_10, train_loader, val_loader)

#Creating class for dropout regularization with dropout rate at p = 0.3
class NetResDeepDropout(nn.Module):
    def __init__(self, n_chans1=32, n_blocks=10):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.resblocks = nn.Sequential(
            *(n_blocks * [ResBlock(n_chans=n_chans1)]))
        self.fc1 = nn.Linear(8 * 8 * n_chans1, 32)
        self.fc2 = nn.Linear(32, 10)
        self.dropout2d = nn.Dropout2d(p=0.3)
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = self.dropout2d(out)
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)
        out = self.dropout2d(out)
        out = out.view(-1, 8 * 8 * self.n_chans1)
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

#Complete CNN with dropout
ResNet_10 = NetResDeepDropout(n_chans1=32, n_blocks=10).to(device=device)
optimizer = optim.SGD(ResNet_10.parameters(), lr=9e-3)
criterion = nn.CrossEntropyLoss() # loss function

# Commented out IPython magic to ensure Python compatibility.
# %%time
# #Doing training loop for dropout
# training_loop(
#     n_epochs = 300, #Needs to be 300
#     optimizer = optimizer,
#     model = ResNet_10,
#     loss_fn = criterion,
#     train_loader = train_loader,
# )
# all_acc_dict["dropout"] = validate(ResNet_10, train_loader, val_loader)

#Creating class to introduce batch normalization as a regularization implementation for CNN model

class NetResDeepBatchNorm(nn.Module):
    def __init__(self, n_chans1=32, n_blocks=10):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv1_norm = nn.BatchNorm2d(n_chans1)
        self.resblocks = nn.Sequential(
            *(n_blocks * [ResBlock(n_chans=n_chans1)]))
        self.fc1 = nn.Linear(8 * 8 * n_chans1, 32)
        self.fc1_norm = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 10)
        
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1_norm(self.conv1(x))), 2)
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 8 * 8 * self.n_chans1)
        out = torch.relu(self.fc1_norm(self.fc1(out)))
        out = self.fc2(out)
        return out

#Complete CNN with batch normalization
ResNet_10 = NetResDeepBatchNorm(n_chans1=32, n_blocks=10).to(device=device)
optimizer = optim.SGD(ResNet_10.parameters(), lr=3e-3)
criterion = nn.CrossEntropyLoss() # loss function

# Commented out IPython magic to ensure Python compatibility.
# %%time
# #Doing training loop for batch normalization
# training_loop(
#     n_epochs = 300, #Needs to be 300
#     optimizer = optimizer,
#     model = ResNet_10,
#     loss_fn = criterion,
#     train_loader = train_loader,
# )
# all_acc_dict["batch_norm"] = validate(ResNet_10, train_loader, valid_loader)