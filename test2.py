import numpy as np
import torch
import torch.nn as nn
import  torch.nn.functional as F
from pt_utils import *


class decom_loss(nn.Module):
    def __init__(self):
        super(decom_loss,self).__init__()

    def forward(self,input_low,input_high,R_low,I_low,R_high,I_high) :
        input_low = torch.FloatTensor(input_low)
        input_high = torch.FloatTensor(input_high)
        R_low = torch.FloatTensor(R_low)
        I_low = torch.FloatTensor(I_low)
        R_high = torch.FloatTensor(R_high)
        I_high = torch.FloatTensor(I_high)

        I_low_3 = torch.cat([I_low, I_low, I_low], axis=1)  # torch.cat
        I_high_3 = torch.cat([I_high, I_high, I_high], axis=1)  # torch.cat

        output_R_low = R_low
        output_R_high = R_high
        output_I_low = I_low_3
        output_I_high = I_high_3
        recon_loss_low = torch.mean(torch.abs(R_low * I_low_3 - input_low))
        recon_loss_high = torch.mean(torch.abs(R_high * I_high_3 - input_high))
        equal_R_loss = torch.mean(torch.abs(R_low - R_high))
        i_mutual_loss = mutual_i_loss2(I_low, I_high)

        i_input_mutual_loss_high = mutual_i_input_loss2(I_high, input_high)
        i_input_mutual_loss_low = mutual_i_input_loss2(I_low, input_low)
        loss_Decom = 1 * recon_loss_high + 1 * recon_loss_low \
                     + 0.01 * equal_R_loss + 0.2 * i_mutual_loss \
                     + 0.15 * i_input_mutual_loss_high + 0.15 * i_input_mutual_loss_low
        return loss_Decom

#tensorflow
class decom_loss2(nn.Module):
    def __init__(self):
        super(decom_loss,self).__init__()

    def forward(self,input_low,input_high,R_low,I_low,R_high,I_high) :
        input_low = torch.FloatTensor(input_low)
        input_high = torch.FloatTensor(input_high)
        R_low = torch.FloatTensor(R_low)
        I_low = torch.FloatTensor(I_low)
        R_high = torch.FloatTensor(R_high)
        I_high = torch.FloatTensor(I_high)

        I_low_3 = torch.cat([I_low, I_low, I_low], axis=1)#torch.cat
        I_high_3 = torch.cat([I_high, I_high, I_high], axis=1)#torch.cat

        output_R_low = R_low
        output_R_high = R_high
        output_I_low = I_low_3
        output_I_high = I_high_3
        recon_loss_low = torch.mean(torch.abs(R_low * I_low_3 - input_low))
        recon_loss_high = torch.mean(torch.abs(R_high * I_high_3 - input_high))

        equal_R_loss = torch.mean(torch.abs(R_low - R_high))

        i_mutual_loss = mutual_i_loss2(I_low, I_high)

        i_input_mutual_loss_high = mutual_i_input_loss2(I_high, input_high)
        i_input_mutual_loss_low = mutual_i_input_loss2(I_low, input_low)
        loss_Decom = 1 * recon_loss_high + 1 * recon_loss_low \
                     + 0.01 * equal_R_loss + 0.2 * i_mutual_loss \
                     + 0.15 * i_input_mutual_loss_high + 0.15 * i_input_mutual_loss_low
        return loss_Decom

mm = decom_loss()
#mm2 = decom_loss2()
R_low = np.ones((2,3,10,10), dtype="float32")
I_low = np.ones((2,1,10,10), dtype="float32")
R_high = np.ones((2,3,10,10), dtype="float32")
I_high  = np.ones((2,1,10,10), dtype="float32")

batch_input_low =np.ones((2,3,10,10), dtype="float32")
batch_input_high = np.ones((2,3,10,10), dtype="float32")


dloss = mm(batch_input_low, batch_input_high, R_low,I_low,R_high,I_high)
#dloss2 = mm2(batch_input_low, batch_input_high, R_low,I_low,R_high,I_high)
print('loss' ,dloss)
#print('loss2' ,dloss2)






