import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import cv2
import numpy as np
from torch.autograd import Variable
import glob
import os

import torch
import torch.nn as nn
import  torch.nn.functional as F



a1 = np.random.rand(1, 2,2,1).astype('float32')
#print('a',a1)
a2 =  np.reshape(a1,[1,1,2,2])
#print('a1',a2)

#a2 =  np.random.rand(1, 1,4,4).astype('float32')
#gradient(a,'x')

def gradient2(input_tensor, direction):
    #input_tensor = torch.FloatTensor(input_tensor)
    #print('input_tensor shape',input_tensor.shape)
    a = input_tensor.shape[0]
    
    b = torch.zeros(input_tensor.shape[2],1)
    b = torch.zeros(input_tensor.shape[0],input_tensor.shape[1],input_tensor.shape[2],1)
    b = b.cuda()
    
    #b = b.unsqueeze(0).unsqueeze(0)
    #print('b shape:',b.shape)
    #print('B',a)
    input_tensor = torch.cat((input_tensor,b),3)
   
    #print('after cat input_tensor', input_tensor.shape)
    a = torch.zeros(1, input_tensor.shape[3])
    a = torch.zeros(input_tensor.shape[0],input_tensor.shape[1],1,input_tensor.shape[3])
    a = a.cuda()
    
    #a = a.unsqueeze(0).unsqueeze(0)
    #print('a', a.shape)
    input_tensor = torch.cat((input_tensor,a), 2)
  
    #print('input_tensor 2', input_tensor.shape)
    c = [[0, 0], [-1, 1]]
    c = torch.FloatTensor(c)
    c = c.cuda()
  
    #nn.init.constant(a,[[0, 0], [-1, 1]])
    # smooth_kernel_x = torch.reshape(nn.init.constant([[0, 0], [-1, 1]], torch.float32), (2, 2, 1))#torch.reshape()
    smooth_kernel_x = torch.reshape(c,(1,1,2,2))# unsqueeze()
    smooth_kernel_y = smooth_kernel_x.permute( [0, 1,3,2])

    #print('gradient_orig:', smooth_kernel_y)
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    weight = nn.Parameter(data=kernel, requires_grad=False)
    gradient_orig = torch.abs(F.conv2d(input_tensor, weight,stride=1,padding=0 ))
    
    #c = gradient_orig
    #print('c shape',c.shape)
    #c = c.permute([0,2,3,1]).cpu().detach().numpy()
    #print('c shape',c[0])
    #cv2.imwrite('./gradient.jpg',c[0]*255)

    grad_min = torch.min(gradient_orig)#https://blog.csdn.net/devil_son1234/article/details/105542067  torch.min
    grad_max = torch.max(gradient_orig)#torch.max()
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))#torch.div

    #print('pt grad norm',grad_norm)




    #c.weight = kernel
    #gradient_orig = c(input_tensor)
    #print('pt conv:',gradient_orig)
    #print('pt conv shape:', gradient_orig.shape)
    #print('smooth_kernel_x:',smooth_kernel_x)
    #print('smooth2:', tf.constant([[0, 0], [-1, 1]]))
    #smooth_kernel_y = tf.transpose(smooth_kernel_x, [1, 0, 2, 3])#torch.transpose()  

    return grad_norm
#gradient(a1,'y')
#gradient2(a2,'y')


#pad = nn.ZeroPad2d(padding=( 1,  1))
#y = pad(a2)
#print('padding:',y)



#pytorch

def mutual_i_loss2(input_I_low, input_I_high):#loss
    
    low_gradient_x = gradient2(input_I_low, "x")
    
    high_gradient_x = gradient2(input_I_high, "x")

    x_loss = torch.mul((low_gradient_x + high_gradient_x),torch.exp(-10*(low_gradient_x+high_gradient_x)))
    low_gradient_y = gradient2(input_I_low, "y")
    high_gradient_y = gradient2(input_I_high, "y")
    y_loss = torch.mul((low_gradient_y + high_gradient_y),torch.exp(-10*(low_gradient_y+high_gradient_y)))# torch.exp()
    mutual_loss = torch.mean( x_loss + y_loss)  #torch.min
    #print('pytorch mutual_loss', mutual_loss)
    return mutual_loss

low = np.random.rand(1, 2,2,1).astype('float32')
low2 =  np.reshape(low,[1,1,2,2])

high = np.random.rand(1, 2,2,3).astype('float32')
high2 =  torch.FloatTensor(high).permute([0,3,1,2])




class RGB2Gray(nn.Module):
    def __init__(self):
        super(RGB2Gray, self).__init__()
        _kernel = [0.299, 0.587, 0.114]#[0.2125, 0.7154, 0.0721]
        _kernel = torch.tensor(_kernel).view(1, 3, 1, 1)
        self.weight = _kernel.cuda()

    def forward(self, x):
        #print('weight pos:',self.weight.device)#cpu
        gray =  F.conv2d(x, self.weight)
        return gray





#pytorch
def mutual_i_input_loss2(input_I_low, input_im):    #loss 
    #input_gray = tf.image.rgb_to_grayscale(input_im)#
    gray = RGB2Gray().cuda()
    #print('gray pos:',gray.device)#cpu
    input_gray = gray(input_im)

    low_gradient_x = gradient2(input_I_low, "x")
    input_gradient_x = gradient2(input_gray, "x")
    b = [0.01]
    b = torch.Tensor(b).cuda()
    x_loss = torch.abs(torch.div(low_gradient_x, torch.max(input_gradient_x, b)))#torch.max()
    #print('torch.max',torch.max(input_gradient_x, b))
    low_gradient_y = gradient2(input_I_low, "y")
    input_gradient_y = gradient2(input_gray, "y")
    y_loss = torch.abs(torch.div(low_gradient_y, torch.max(input_gradient_y, b)))#torch.max()
    mut_loss = torch.mean(x_loss + y_loss) #torch.mean
    #print('pytorch:',mut_loss)
    return mut_loss


