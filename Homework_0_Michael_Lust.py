#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Michael Lust
#801094861
#Real Time AI (4106)
#January 26, 2022


# In[2]:


from torchvision import models, datasets, transforms
import torch
import torch.nn as nn


# In[3]:


dir(models)


# In[4]:


alexnet = models.AlexNet()


# In[5]:


#Using ResNet 101
resnet = models.resnet101(pretrained=True)


# In[6]:


resnet


# In[7]:


#Using ResNetGen

class ResNetBlock(nn.Module): # <1>

    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x) # <2>
        return out


class ResNetGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9): # <3> 

        assert(n_blocks >= 0)
        super(ResNetGenerator, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=True),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResNetBlock(ngf * mult)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input): # <3>
        return self.model(input)


# In[8]:


netG = ResNetGenerator()


# In[9]:


netG


# In[10]:


from torchvision import transforms
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])


# In[11]:


#Problem 1 using ResNet 101


# In[12]:


from PIL import Image
img = Image.open("Photos/Tree.jpg")


# In[13]:


img


# In[14]:


img_t = preprocess(img)


# In[15]:


batch_t = torch.unsqueeze(img_t, 0)


# In[16]:


resnet.eval()


# In[17]:


out = resnet(batch_t)
out


# In[18]:


with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]


# In[19]:


_, index = torch.max(out, 1)


# In[20]:


percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
labels[index[0]], percentage[index[0]].item()


# In[21]:


_, indices = torch.sort(out, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]


# In[22]:


#Problem 3 Part 1 Model Complexity of ResNet 101


# In[23]:


from ptflops import get_model_complexity_info

macs, params = get_model_complexity_info(resnet, (3, 256, 256), as_strings=True, print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[24]:


#Problem 2 using ResnetGen


# In[25]:


model_path = 'horse2zebra_0.4.0.pth'
model_data = torch.load(model_path)
netG.load_state_dict(model_data)


# In[26]:


netG.eval()


# In[27]:


preprocess = transforms.Compose([transforms.Resize(256),
                                 transforms.ToTensor()])


# In[28]:


img = Image.open("Photos/Horses.jpg")


# In[29]:


img


# In[30]:


img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)


# In[31]:


batch_out = netG(batch_t)


# In[32]:


out_t = (batch_out.data.squeeze() + 1.0) / 2.0
out_img = transforms.ToPILImage()(out_t)
out_img


# In[33]:


#Problem 3 Part 2 Model complexity of ResnetGen


# In[34]:


macs, params = get_model_complexity_info(netG, (3, 256, 256), as_strings=True, print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[83]:


#Problem 4 Part 1 The MobileNet v2 Architecture 


# In[84]:


model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()


# In[85]:


img = Image.open('Photos/Tree.jpg')
img


# In[86]:


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model


# In[87]:


# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)


# In[88]:


# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


# In[89]:


#Problem 4 Part 2 Model Complexity of the MobileNet v2 architecture 


# In[90]:


macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[ ]:




