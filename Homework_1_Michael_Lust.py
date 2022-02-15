#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Michael Lust
#801094861
#Real Time AI (4106)
#Febuary 14, 2022


# In[2]:


from torchvision import models, datasets, transforms
import imageio as img
import os
import numpy as np
import torch
import torch.nn as nn


# In[3]:


#Problem 1 Using Tensor and .mean() method on RGB pictures


# In[4]:


#Problem 1 part a


# In[5]:


#Loading a picutre with imageio as img and looking at the pixel shape
picture_array = img.imread('Photos/Packet.jpg')
picture_array.shape


# In[6]:


#Problem 1 part b


# In[7]:


#picture = torch.from_numpy(picture_array) 
#ValueError: At least one stride in the given numpy array is negative, 
#and tensors with negative strides are not currently supported. 
#(You can probably work around this by making a copy of your array  with array.copy().) 


picture = torch.from_numpy(np.flip(picture_array,axis=0).copy())
out = picture.permute(2,0,1)


# In[8]:


#3 photos per batch
batch_size = 3 
#We have a batch size of 3 and the pictures are all 4000 H, 3000 W pixels
batch = torch.zeros(batch_size, 3, 4000, 3000, dtype=torch.uint8) 


# In[9]:


#Used the enumerate method to load all .jpg files into a tensor from our Photo's directory to keep track of # of iterations.
picture_directory = 'Photos'
filenames = ['Raycon.jpg', 'Box.jpg', 'Packet.jpg']
for i, filename in enumerate(filenames):
    picture_array = img.imread(os.path.join(picture_directory, filename))
    picture_t = torch.from_numpy(np.flip(picture_array,axis=0).copy())
    picture_t = picture_t.permute(2,0,1)
    picture_t = picture_t[:3]
    batch[i] = picture_t 
    print(batch[i])


# In[10]:


#Problem 1 part c 


# In[11]:


#mean(): input dtype should be either floating point or complex dtypes.
batch_norm = batch.float()
batch_mean = batch.float()
#Normalizing the data by divide the values of the pixels by 255 (the maximum representable number in 8-bit unsigned)
batch_norm /= 255.0
batch_mean /= 255.0
batch_norm


# In[12]:


#Using standard deviation method to normalize the data from -1 to 1 for better neural network performance.
n_channels_norm = batch_norm.shape[1]
n_channels_mean = batch_mean.shape[1]
#Computing just the mean of each channel of my images.
for c in range(n_channels_norm):
    mean = torch.mean(batch_norm[:, c])
    std = torch.std(batch_norm[:, c])
    batch_norm[:, c] = (batch_norm[:, c] - mean) / std
    print(batch_norm[:, c])
    


# In[13]:


#Calculating just the mean or abunce of Red in channel 1, Blue in channel 2, and Green in channel 3
for c in range(n_channels_mean):
    mean = torch.mean(batch_mean[:, c])
    print('Channel Averages for each picture', mean)   


# In[14]:


#Problem 2 Changing temperature prediction model example from lecture 5 to an non-linear model to compare


# In[15]:


#Recreating the temperature prediction model 


# In[16]:


t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)


# In[17]:


def model(t_u, w, b):
    equ = w * t_u + b 
    return equ


# In[18]:


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


# In[19]:


w = torch.ones(()) #initial W is 1
b = torch.zeros(()) #initial b is 0
t_p = model(t_u, w, b)
t_p


# In[20]:


loss = loss_fn(t_p, t_c)
loss


# In[21]:


delta = 0.1
loss_rate_of_change_w = (loss_fn(model(t_u, w + delta, b), t_c) - 
                         loss_fn(model(t_u, w - delta, b), t_c)) / (2.0 * delta)
loss_rate_of_change_w 


# In[22]:


a = 1e-2     #means 0.01 is the learning rate or the changing rate for our parameters.
w = w - a * loss_rate_of_change_w
w


# In[23]:


loss_rate_of_change_b = (loss_fn(model(t_u, w, b + delta), t_c) - 
                         loss_fn(model(t_u, w, b - delta), t_c)) / (2.0 * delta)
b = b - a * loss_rate_of_change_b
b


# In[24]:


def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)
    return dsq_diffs


# In[25]:


def dmodel_dw(t_u, w, b):
    return t_u


# In[26]:


def dmodel_db(t_u, w, b):
    return 1.0


# In[27]:


def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])


# In[28]:


#Modified training loop to return Epoch and Cost Values to graph
def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    
    loss_arr = []
    n_epochs_arr = []
    
    for epoch in range(1, n_epochs + 1):
        w, b = params
        
        t_p = model(t_u, w, b)
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w, b)
        params = params - learning_rate * grad
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
        
        loss_arr.append([loss])
        n_epochs_arr.append([epoch])
        
    return params, loss_arr, n_epochs_arr


# In[29]:


#Creating the training loop for temperature prediction
#Normalizing the inputs between -1 to 1.
t_un = 0.1 * t_u

#Training with 5000 epochs and learning rate at 1e-1
training_loop(
n_epochs = 5000,
learning_rate = 1e-1,
params = torch.tensor([1.0, 0.0]),
t_u = t_un,
t_c = t_c)


# In[30]:


#Training with 5000 epochs and learning rate at 1e-2
params, loss_arr, n_epochs_arr = training_loop(
n_epochs = 5000,
learning_rate = 1e-2,
params = torch.tensor([1.0, 0.0]),
t_u = t_un,
t_c = t_c)


# In[31]:


#Training with 5000 epochs and learning rate at 1e-3
training_loop(
n_epochs = 5000,
learning_rate = 1e-3,
params = torch.tensor([1.0, 0.0]),
t_u = t_un,
t_c = t_c)


# In[32]:


#Training with 5000 epochs and learning rate at 1e-4
training_loop(
n_epochs = 5000,
learning_rate = 1e-4,
params = torch.tensor([1.0, 0.0]),
t_u = t_un,
t_c = t_c)


# In[33]:


from matplotlib import pyplot as plt
t_p = model(t_un, *params)


# In[34]:


fig = plt.figure(dpi = 600)
plt.xlabel("Temperature (Fahrenheit)")
plt.ylabel("Temperature (Celsius)")
plt.plot(t_u.numpy(), t_p.detach() .numpy())
plt.plot(t_u.numpy(), t_c.numpy(), 'o')


# In[35]:


#Plotting the Loss over the Number of Epochs for all X values combined
plt.plot(n_epochs_arr, loss_arr, color='Blue', label='Loss Line', linewidth = 3 )
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of Epochs')
plt.ylabel('Value for Cost')
plt.title('Convergence for Gradient Descent for Housing Prices base on Various Inputs')
plt.legend()
plt.show()


# In[36]:


#Problem 2 Part a Changing model to a non-linear model


# In[37]:


#Recreating the temperature prediction model 
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)


# In[38]:


def model(t_u, w1, w2, b):
    equ = w2 * t_u ** 2 + w1 * t_u + b 
    return equ


# In[39]:


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


# In[40]:


w1 = torch.ones(()) #initial W1 is 1
w2 = torch.ones(()) #initial W2 is 1
b = torch.zeros(()) #initial b is 0
t_p = model(t_u, w1, w2, b)
t_p


# In[41]:


loss = loss_fn(t_p, t_c)
loss


# In[42]:


delta = 0.1
loss_rate_of_change_w = (loss_fn(model(t_u, w1 + delta, w2 + delta, b), t_c) - 
                         loss_fn(model(t_u, w1 - delta, w2 - delta, b), t_c)) / (2.0 * delta)
loss_rate_of_change_w


# In[43]:


a = 1e-2     #means 0.01 is the learning rate or the changing rate for our parameters.
w1 = w1 - a * loss_rate_of_change_w
w1


# In[44]:


w2 = w2 - a * loss_rate_of_change_w
w2


# In[45]:


loss_rate_of_change_b = (loss_fn(model(t_u, w1, w2, b + delta), t_c) - 
                         loss_fn(model(t_u, w1, w2, b - delta), t_c)) / (2.0 * delta)
b = b - a * loss_rate_of_change_b
b


# In[46]:


def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)
    return dsq_diffs


# In[47]:


def dmodel_dw1(t_u, w1, w2, b):
    return t_u


# In[48]:


def dmodel_dw2(t_u, w1, w2, b):
    return t_u


# In[49]:


def dmodel_db(t_u, w1, w2, b):
    return 1.0


# In[50]:


def grad_fn(t_u, t_c, t_p, w1, w2, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw1 = dloss_dtp * dmodel_dw1(t_u, w1, w2, b)
    dloss_dw2 = dloss_dtp * dmodel_dw2(t_u, w1, w2, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w1, w2, b)
    return torch.stack([dloss_dw1.sum(), dloss_dw2.sum(), dloss_db.sum()])


# In[51]:


#Modified training loop to return Epoch and Cost Values to graph
def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    loss_arr = []
    n_epochs_arr = []
    
    for epoch in range(1, n_epochs + 1):
        w1, w2, b = params
        
        t_p = model(t_u, w1, w2, b)
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w1, w2, b)
        params = params - learning_rate * grad
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
        
        loss_arr.append([loss])
        n_epochs_arr.append([epoch])
        
    return params, loss_arr, n_epochs_arr


# In[52]:


#Problem 2 Part b


# In[53]:


#Creating the training loop for temperature prediction
#Normalizing the inputs between -1 to 1.
t_un = 0.1 * t_u

#Training with 5000 epochs and learning rate at 1e-1
training_loop(
n_epochs = 5000,
learning_rate = 1e-1,
params = torch.tensor([1.0, 1.0, 0.0]),
t_u = t_un,
t_c = t_c)


# In[54]:


#Training with 5000 epochs and learning rate at 1e-2
training_loop(
n_epochs = 5000,
learning_rate = 1e-2,
params = torch.tensor([1.0, 1.0, 0.0]),
t_u = t_un,
t_c = t_c)


# In[55]:


#Training with 5000 epochs and learning rate at 1e-3
params, loss_arr, n_epochs_arr = training_loop(
n_epochs = 5000,
learning_rate = 1e-3,
params = torch.tensor([1.0, 1.0, 0.0]),
t_u = t_un,
t_c = t_c)


# In[56]:


#Training with 5000 epochs and learning rate at 1e-4
training_loop(
n_epochs = 5000,
learning_rate = 1e-4,
params = torch.tensor([1.0, 1.0, 0.0]),
t_u = t_un,
t_c = t_c)


# In[57]:


t_p = model(t_un, *params)


# In[58]:


fig = plt.figure(dpi = 100)
plt.xlabel("Temperature (Fahrenheit)")
plt.ylabel("Temperature (Celsius)")
plt.plot(t_u.numpy(), t_p.detach() .numpy())
plt.plot(t_u.numpy(), t_c.numpy(), 'o')


# In[59]:


#Plotting the Loss over the Number of Epochs for all X values combined
plt.plot(n_epochs_arr, loss_arr, color='Blue', label='Loss Line', linewidth = 3 )
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of Epochs')
plt.ylabel('Value for Cost')
plt.title('Convergence for Gradient Descent for Housing Prices base on Various Inputs')
plt.legend()
plt.show()


# In[60]:


#Problem 3 Developing model to predict housing prices


# In[61]:


#Problem 3 part a Setting up model and parameters


# In[62]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt  


# In[63]:


dataset = pd.DataFrame(pd.read_csv('Housing.csv'))
dataset.head()


# In[64]:


m = len(dataset)
m


# In[65]:


dataset.shape


# In[66]:


num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price'] 
Newtrain = dataset[num_vars] 
Newtrain.head() 


# In[67]:


Newtrain.values[:, 0]


# In[68]:


area_t_u = torch.tensor(Newtrain.values[:, 0])
bedrooms_t_u = torch.tensor(Newtrain.values[:, 1])
bathrooms_t_u = torch.tensor(Newtrain.values[:, 2])
stories_t_u = torch.tensor(Newtrain.values[:, 3])
parking_t_u = torch.tensor(Newtrain.values[:, 4])

#Predicting the housing prices
price_t_c = torch.tensor(Newtrain.values[:, 5])


# In[69]:


def model(w1, w2, w3, w4, w5, x1, x2, x3, x4, x5, b):
    equ = w5*x5 + w4*x4 + w3*x3 + w2*x2 + w1*x1 + b 
    return equ


# In[70]:


x1 = area_t_u
x2 = bedrooms_t_u
x3 = bathrooms_t_u 
x4 = stories_t_u 
x5 = parking_t_u 


# In[71]:


#All below needs work


# In[72]:


def loss_fn(t_p, price_t_c):
    squared_diffs = (t_p - price_t_c)**2
    return squared_diffs.mean()


# In[73]:


w1 = torch.ones(()) #initial W is 1
w2 = torch.ones(()) #initial W is 1
w3 = torch.ones(()) #initial W is 1
w4 = torch.ones(()) #initial W is 1
w5 = torch.ones(()) #initial W is 1
b = torch.zeros(()) #initial b is 0
t_p = model(w1, w2, w3, w4, w5, x1, x2, x3, x4, x5, b)
t_p


# In[74]:


loss = loss_fn(t_p, price_t_c)
loss


# In[75]:


delta = 0.1
loss_rate_of_change_w = (loss_fn(model(w1 + delta, w2 + delta, w3 + delta, w4+ delta, w5+ delta, x1, x2, x3, x4, x5, b), price_t_c) - 
                         loss_fn(model(w1 - delta, w2 - delta, w3 - delta, w4 - delta, w5 - delta, x1, x2, x3, x4, x5, b), price_t_c)) / (2.0 * delta)
loss_rate_of_change_w 


# In[76]:


a = 1e-2     #means 0.01 is the learning rate or the changing rate for our parameters.
w = w - a * loss_rate_of_change_w
w


# In[77]:


loss_rate_of_change_b = (loss_fn(model(w1, w2, w3, w4, w5, x1, x2, x3, x4, x5, b + delta), price_t_c) - loss_fn(model(w1, w2, w3, w4, w5, x1, x2, x3, x4, x5, b - delta), price_t_c)) / (2.0 * delta)
b = b - a * loss_rate_of_change_b
b


# In[78]:


def dloss_fn(t_p, price_t_c):
    dsq_diffs = 2 * (t_p - price_t_c) / t_p.size(0)
    return dsq_diffs


# In[79]:


def dmodel_dw_w1(w1, w2, w3, w4, w5, x1, x2, x3, x4, x5, b):
    return x1


# In[80]:


def dmodel_dw_w2(w1, w2, w3, w4, w5, x1, x2, x3, x4, x5, b):
    return x2


# In[81]:


def dmodel_dw_w3(w1, w2, w3, w4, w5, x1, x2, x3, x4, x5, b):
    return x3


# In[82]:


def dmodel_dw_w4(w1, w2, w3, w4, w5, x1, x2, x3, x4, x5, b):
    return x4


# In[83]:


def dmodel_dw_w5(w1, w2, w3, w4, w5, x1, x2, x3, x4, x5, b):
    return x5


# In[84]:


def dmodel_db(w1, w2, w3, w4, w5, x1, x2, x3, x4, x5, b):
    return 1.0


# In[85]:


def grad_fn(price_t_c, t_p, w1, w2, w3, w4, w5, x1, x2, x3, x4, x5, b):
    dloss_dtp = dloss_fn(t_p, price_t_c)
    dloss_dw_w1 = dloss_dtp * dmodel_dw_w1(w1, w2, w3, w4, w5, x1, x2, x3, x4, x5, b)
    dloss_dw_w2 = dloss_dtp * dmodel_dw_w2(w1, w2, w3, w4, w5, x1, x2, x3, x4, x5, b)
    dloss_dw_w3 = dloss_dtp * dmodel_dw_w3(w1, w2, w3, w4, w5, x1, x2, x3, x4, x5, b)
    dloss_dw_w4 = dloss_dtp * dmodel_dw_w4(w1, w2, w3, w4, w5, x1, x2, x3, x4, x5, b)
    dloss_dw_w5 = dloss_dtp * dmodel_dw_w5(w1, w2, w3, w4, w5, x1, x2, x3, x4, x5, b)
    dloss_db = dloss_dtp * dmodel_db(w1, w2, w3, w4, w5, x1, x2, x3, x4, x5, b)
    return torch.stack([dloss_dw_w1.sum(), dloss_dw_w2.sum(), dloss_dw_w3.sum(), dloss_dw_w4.sum(), dloss_dw_w5.sum(), dloss_db.sum()])


# In[86]:


#Modified training loop to return Epoch and Cost Values to graph
def training_loop(n_epochs, learning_rate, params, x1, x2, x3, x4, x5, price_t_c):
    loss_arr = []
    n_epochs_arr = []
    
    for epoch in range(1, n_epochs + 1):
        w1, w2, w3, w4, w5, b = params
        
        t_p = model(w1, w2, w3, w4, w5, x1, x2, x3, x4, x5, b)
        loss = loss_fn(t_p, price_t_c)
        grad = grad_fn(price_t_c, t_p, w1, w2, w3, w4, w5, x1, x2, x3, x4, x5, b)
        params = params - learning_rate * grad
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
        
        loss_arr.append([loss])
        n_epochs_arr.append([epoch])
        
    return params, loss_arr, n_epochs_arr


# In[87]:


x2


# In[88]:


#Using Standard Scalar build in API to better train the neural network model
#Changed the explanatory values within a mean of 0.
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#define standard scalar
scalar = StandardScaler()
#scalar = MinMaxScaler()

#Scaling from Newtrain[num_vars]
x1_un = scalar.fit_transform(Newtrain[['area']])
x2_un = scalar.fit_transform(Newtrain[['bedrooms']])
x3_un = scalar.fit_transform(Newtrain[['bathrooms']])
x4_un = scalar.fit_transform(Newtrain[['stories']])
x5_un = scalar.fit_transform(Newtrain[['parking']])
price_t_c_un = scalar.fit_transform(Newtrain[['price']])

x1_un = torch.tensor(x1_un)
x2_un = torch.tensor(x2_un)
x3_un = torch.tensor(x3_un)
x4_un = torch.tensor(x4_un)
x5_un = torch.tensor(x5_un)
price_t_c_un = torch.tensor(price_t_c_un)


# In[89]:


x1_un_torch = torch.reshape(x1_un, (-1,))
x2_un_torch = torch.reshape(x2_un, (-1,))  
x3_un_torch = torch.reshape(x3_un, (-1,))  
x4_un_torch = torch.reshape(x4_un, (-1,))  
x5_un_torch = torch.reshape(x5_un, (-1,))
price_t_c_torch = torch.reshape(price_t_c_un, (-1,))

x2_un_torch


# In[90]:


#Doing the training loop for different learning rates

#Normalizing the inputs between -1 to 1.
#x1_un = 0.1 * x1
#x2_un = 0.1 * x2
#x3_un = 0.1 * x3
#x4_un = 0.1 * x4
#x5_un = 0.1 * x5

#Learning rate is at 1e-1 with 5000 epochs
training_loop(
n_epochs = 5000,
learning_rate = 1e-1,
params = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
x1 = x1_un_torch,
x2 = x2_un_torch,
x3 = x3_un_torch,
x4 = x4_un_torch,
x5 = x5_un_torch,
price_t_c = price_t_c_torch)


# In[91]:


#Learning rate is at 1e-2 with 5000 epochs
training_loop(
n_epochs = 5000,
learning_rate = 1e-2,
params = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
x1 = x1_un_torch,
x2 = x2_un_torch,
x3 = x3_un_torch,
x4 = x4_un_torch,
x5 = x5_un_torch,
price_t_c = price_t_c_torch)


# In[92]:


#Learning rate is at 1e-3 with 5000 epochs
params, loss_arr, n_epochs_arr = training_loop(
n_epochs = 5000,
learning_rate = 1e-3,
params = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
x1 = x1_un_torch,
x2 = x2_un_torch,
x3 = x3_un_torch,
x4 = x4_un_torch,
x5 = x5_un_torch,
price_t_c = price_t_c_torch)


# In[93]:


#Learning rate is at 1e-3 with 5000 epochs
training_loop(
n_epochs = 5000,
learning_rate = 1e-4,
params = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
x1 = x1_un_torch,
x2 = x2_un_torch,
x3 = x3_un_torch,
x4 = x4_un_torch,
x5 = x5_un_torch,
price_t_c = price_t_c_torch)
#print_params = False)
params


# In[94]:


loss_arr


# In[95]:


#Plotting the Loss over the Number of Epochs for all X values combined
plt.plot(n_epochs_arr, loss_arr, color='Blue', label='Loss Line', linewidth = 3 )
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of Epochs')
plt.ylabel('Value for Cost')
plt.title('Convergence for Gradient Descent for Housing Prices base on Various Inputs')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




