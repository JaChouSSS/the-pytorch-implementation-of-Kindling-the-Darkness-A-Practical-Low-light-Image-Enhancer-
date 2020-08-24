import random
from model3 import *
from pt_utils import *
import torch.optim as optim
import os
import torch
import glob
import cv2
import time
from torch.utils.data import Dataset, DataLoader
from padding_same_conv import *


#########################  decomnet


#########################

def pt_grad_loss(input_r_low, input_r_high):
    #input_r_low = torch.Tensor(input_r_low)
    #input_r_high = torch.Tensor(input_r_high)
    gray = RGB2Gray().cuda()
    input_r_low_gray = gray(input_r_low)
    input_r_high_gray = gray(input_r_high)
    #input_r_low_gray = tf.image.rgb_to_grayscale(input_r_low)
    #input_r_high_gray = tf.image.rgb_to_grayscale(input_r_high)
    #t= gradient2(input_r_low_gray, 'x')
    #print('t ===== ',t)
    #print('input_r_low_gray',input_r_low_gray.shape)
    x_loss = torch.pow(gradient2(input_r_low_gray, 'x') - gradient2(input_r_high_gray, 'x'),2)
    #print('pt x_loss ',x_loss.shape)
    y_loss = torch.pow(gradient2(input_r_low_gray, 'y') - gradient2(input_r_high_gray, 'y'),2)
    grad_loss_all = torch.mean(x_loss + y_loss)
    #print('pt grad_loss ', grad_loss_all)
    return grad_loss_all
def gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    #print('x_data :',x_data.shape)

    #x_data = np.expand_dims(x_data, axis=-1)
    #x_data = np.expand_dims(x_data, axis=-1)
    x_data = torch.Tensor(x_data).unsqueeze(0).unsqueeze(0)

    #print('x_data :', y_data)

    y_data =  torch.Tensor(y_data).unsqueeze(0).unsqueeze(0)

    #y_data = np.expand_dims(y_data, axis=-1)
    #y_data = np.expand_dims(y_data, axis=-1)
    #print('x_data2 :', x_data.shape)
    #x = tf.constant(x_data, dtype=tf.float32)
    #y = tf.constant(y_data, dtype=tf.float32)

    #g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    g = torch.exp(-((x_data **2 + y_data **2)/(2.0 * sigma ** 2)))
    return g / torch.sum(g)

def pt_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    #img1 = torch.Tensor(img1)
    #img2 = torch.Tensor(img2)
    window = gauss(size, sigma) # window shape [size, size]
    K1 = torch.Tensor([0.01]).cuda()
    K2 = torch.Tensor([0.03]).cuda()
    L = torch.Tensor([1]).cuda()  # depth of image (255 in case the image has a differnt scale)
    C1 = torch.pow(K1*L,2)
    C2 = torch.pow(K2*L,2)
    #mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    #mu1  = nn.Parameter(data=window, requires_grad=False)
    weight = nn.Parameter(data=window, requires_grad=False).cuda()
    mu1 = F.conv2d(img1,weight).cuda()
    #mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu2 = F.conv2d(img2,weight).cuda()
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    #sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID')
    sigma1_sq = F.conv2d(img1*img1,weight).cuda()- mu1_sq
    #sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma2_sq = F.conv2d(img2*img2,weight).cuda()- mu2_sq
    #sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    sigma12 =  F.conv2d(img1*img2,weight).cuda()- mu1_mu2

    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = torch.mean(value)
    #print('pt ' ,value)
    return value

def pt_ssim_loss(output_r, input_high_r):
    output_r_1 = output_r[:,0:1,:,:]  # R
    input_high_r_1 = input_high_r[:,0:1,:,:]
    ssim_r_1 = pt_ssim(output_r_1, input_high_r_1)
    #print('pt r_1', ssim_r_1)
    output_r_2 = output_r[:,1:2,:,:]   #G
    input_high_r_2 = input_high_r[:,1:2,:,:]
    ssim_r_2 = pt_ssim(output_r_2, input_high_r_2)
    #print('pt r_1', ssim_r_2)
    output_r_3 = output_r[:,2:3,:,:]  #B
    input_high_r_3 = input_high_r[:,2:3,:,:]
    ssim_r_3 = pt_ssim(output_r_3, input_high_r_3)
    #print('pt r_1', ssim_r_3)
    ssim_r = (ssim_r_1 + ssim_r_2 + ssim_r_3)/3.0
    loss_ssim1 = 1-ssim_r
    print('pt ssim loss :', loss_ssim1)
    return loss_ssim1

def grad_loss(input_i_low, input_i_high):
    x_loss = torch.pow(gradient2(input_i_low, 'x') - gradient2(input_i_high, 'x'),2)
    y_loss = torch.pow(gradient2(input_i_low, 'y') - gradient2(input_i_high, 'y'),2)
    grad_loss_all = torch.mean(x_loss + y_loss)
    return grad_loss_all


# def adjust_loss(output_r , input_high_r):
#     loss_ssim = pt_ssim_loss(output_r, input_high_r)
#     loss_grad = pt_grad_loss(output_r, input_high_r)
#     loss_square = torch.mean(tf.square(output_r - input_high_r))  # * ( 1 - input_low_r ))#* (1- input_low_i)))
#     loss_adjust = loss_square + loss_grad + loss_ssim
#     return loss_adjust


class restore_loss(nn.Module):
    def __init__(self):
        super(restore_loss,self).__init__()
    def forward(self,output_r , input_high_r):
        loss_ssim = pt_ssim_loss(output_r, input_high_r)
        loss_grad = pt_grad_loss(output_r, input_high_r)
        loss_square = torch.mean(torch.pow((output_r - input_high_r),2))  # * ( 1 - input_low_r ))#* (1- input_low_i)))
        loss_adjust = loss_square + loss_grad + loss_ssim
        return loss_adjust


def load_images(im):
    img = np.array(im, dtype="float32") / 255.0
    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = np.float32((img - img_min) / np.maximum((img_max - img_min), 0.001))  #
    return img_norm


def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)  #
    # torch.flip
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)  #
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)  #
        return np.flipud(image)  #
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)  #
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)  #
        return np.flipud(image)  #
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)  #
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)  #
        return np.flipud(image)  #


def read_directory(directory_name):
    array_of_img = []
    # print(os.listdir(directory_name))
    for filename in os.listdir(directory_name):
        img = cv2.imread(directory_name + "/" + filename)  # directory_name + "/" +
        img = load_images(img)
        array_of_img.append(img)

        # print(img)
        # print('img:',img)
    return array_of_img


class adjustdataset(Dataset):
    def __init__(self, low_i, low_r, h_r):
        self.low_i = low_i
        self.low_r = low_r
        self.h_r = h_r
        self.len = len(h_r)

    def __getitem__(self, index):
        # self.low_img[index], self.high_img[index]
        #print('low_i shape', self.low_i.shape)
        h = self.low_i[index].shape[0]
        w = self.low_i[index].shape[1]
        #print(' h ', h, ' w ', w)
        x = random.randint(0, h - patch_size)  #
        y = random.randint(0, w - patch_size)  #
        rand_mode = random.randint(0, 7)  #
        #print('low_i shape :', self.low_i[0].shape)
        start = time.time()
        l_i = data_augmentation(
            self.low_i[index][x: x + patch_size, y: y + patch_size, :], rand_mode)
        l_r = data_augmentation(
            self.low_r[index][x: x + patch_size, y: y + patch_size, :], rand_mode)  #
        hh_r = data_augmentation(
            self.h_r[index][x: x + patch_size, y: y + patch_size, :], rand_mode)
        end = time.time()
        #print('data_augment time',end - start)#e-5  ~  e-4
        
        start = time.time()
        l_i = l_i.copy()
        l_r = l_r.copy()
        hh_r = hh_r.copy()
        l_r = torch.tensor(l_r).cuda()
        l_i = torch.tensor(l_i).cuda()
        hh_r = torch.tensor(hh_r).cuda()  # return low, high
        end = time.time()
        print('copy cuda time',end - start) #0.03 ~ 0.005
        #print('l_r shape :', l_r.shape)
        #print('l_i shape :', l_i.shape)
        #print('hh_r shape :', hh_r.shape)
        return l_r, l_i, hh_r

    def __len__(self):
        return self.len


class DealDataset(Dataset):
    def __init__(self):
        # xy = np.loadtxt('../dataSet/diabetes.csv.gz', delimiter=',', dtype=np.float32)  #
        # self.x_data = torch.from_numpy(xy[:, 0:-1])
        # self.y_data = torch.from_numpy(xy[:, [-1]])
        # self.len = xy.shape[0]
        # low = open('')
        # hight = open('')

        # print(self.low_names)

        start = time.time()
        self.low_img = read_directory('/home/intern2/jay/project/pt_kind/dataset/our485/low')
        self.high_img = read_directory('/home/intern2/jay/project/pt_kind/dataset/our485/high')
        self.low_i = []
        self.low_r = []
        self.h_r = []
        end = time.time()
        print('load img time :', end - start)

        # self.low_names.sort()
        # self.high_names.sort()
        # print(self.low_names)

        #
        # im = Image.open(self.low_names[0])
        # im = cv2.imread(im)
        # im = torch.FloatTensor(im)
        # print(im)
        self.len = len(self.low_img)

    def __getitem__(self, index):
        self.low_img[index], self.high_img[index]
        h = self.low_img[index].shape[0]
        w = self.low_img[index].shape[1]
        x = random.randint(0, h - patch_size)  #
        y = random.randint(0, w - patch_size)  #
        # rand_mode = random.randint(0, 7)  #
        # low = data_augmentation(
        #    self.low_img[index][x: x + patch_size, y: y + patch_size, :], rand_mode)
        # high = data_augmentation(
        #    self.high_img[index][x: x + patch_size, y: y + patch_size, :], rand_mode)  #
        # low = low.copy()
        # high = high.copy()

        # low = torch.tensor(low)
        # high = torch.tensor(high)
        # return low, high
        return self.low_img[index], self.high_img[index]

    def __len__(self):
        return self.len


# loss
class decom_loss(nn.Module):
    def __init__(self):
        super(decom_loss, self).__init__()

    def forward(self, input_low, input_high, R_low, I_low, R_high, I_high):  # input_low,input_high,
        # input_low = torch.Tensor(input_low)
        # input_high = torch.Tensor(input_high)
        # R_low = torch.Tensor(R_low)
        # I_low = torch.Tensor(I_low)
        # R_high = torch.Tensor(R_high)
        # I_high = torch.Tensor(I_high)
        # print('enter forward===========================')
        I_low_3 = torch.cat([I_low, I_low, I_low], axis=1)  # torch.cat
        I_high_3 = torch.cat([I_high, I_high, I_high], axis=1)  # torch.cat
        # print('  1  ===========================')
        output_R_low = R_low
        output_R_high = R_high
        output_I_low = I_low_3
        output_I_high = I_high_3
        recon_loss_low = torch.mean(torch.abs(R_low * I_low_3 - input_low))  # torch.mean orch.abs
        recon_loss_high = torch.mean(torch.abs(R_high * I_high_3 - input_high))  # torch.mean orch.abs
        # print('  2  ===========================')
        equal_R_loss = torch.mean(torch.abs(R_low - R_high))
        # print('  3  ===========================')
        i_mutual_loss = mutual_i_loss2(I_low, I_high)
        # print('  4  ===========================')

        i_input_mutual_loss_high = mutual_i_input_loss2(I_high, input_high)
        i_input_mutual_loss_low = mutual_i_input_loss2(I_low, input_low)
        # loss_Decom = 1 * recon_loss_high + 1 * recon_loss_low \
        #             + 0.01 * equal_R_loss + 0.2 * i_mutual_loss \
        #             + 0.15 * i_input_mutual_loss_high + 0.15 * i_input_mutual_loss_low
        # print('loss 1 :',loss_Decom)
        t1 = torch.tensor([1]).cuda()
        t2 = torch.tensor([0.01]).cuda()
        t3 = torch.tensor([0.2]).cuda()  # 0.2
        t4 = torch.tensor([0.02]).cuda()  # 0.15
        t5 = torch.tensor([0.15]).cuda()
        # m0 = torch.mul(recon_loss_high,t1)
        # m1 = torch.mul(recon_loss_low,t1)
        # m2 = torch.mul(equal_R_loss,t2)
        # m3 =  torch.mul(i_mutual_loss,t3)
        # m4 = torch.mul(i_input_mutual_loss_high,t4)
        # m5 =  torch.mul(i_input_mutual_loss_low,t4)
        # loss_Decom = torch.tensor([0])
        # loss_Decom  =torch.add(loss_Decom , m0)
        # loss_Decom = torch.add(loss_Decom, m1)
        # loss_Decom = torch.add(loss_Decom, m2)
        # loss_Decom = torch.add(loss_Decom, m3)
        # loss_Decom = torch.add(loss_Decom, m4)
        # loss_Decom = torch.add(loss_Decom, m5)
        loss_Decom = torch.mul(recon_loss_high, t1) + torch.mul(recon_loss_low, t1) + torch.mul(equal_R_loss, t2) + \
                     torch.mul(i_mutual_loss, t3) + torch.mul(i_input_mutual_loss_high, t4) + torch.mul(
            i_input_mutual_loss_low, t5)  # \
        # torch.mul(i_input_mutual_loss_high,t4) + torch.mul(i_input_mutual_loss_low,t4)
        # print('loss :',loss_Decom)
        # print('loss:',loss_Decom)
        # loss_Decom.requires_grad = True
        # loss_Decom = torch.tensor(loss_Decom)
        return loss_Decom

    # low_img=read_directory('/home/intern2/jay/project/pt_kind/dataset/our485/low')


# high_img = read_directory('/home/intern2/jay/project/pt_kind/dataset/our485/high')

# low_img = torch.tensor(low_img[0:100,:,:,:]).cuda()
# high_img = torch.tensor(high_img).cuda()


def save_images(filepath, result_1, result_2=None, result_3=None):
    result_1 = result_1.cpu().detach().numpy()
    result_2 = result_2.cpu().detach().numpy()
    # print('result1 shape',result_1.shape)
    # print('result2 shape',result_2.shape)
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)
    # result_3 = np.squeeze(result_3)

    # if not result_2.any():
    #    cat_image = result_1
    # else:
    cat_image = np.concatenate([result_1, result_2], axis=1)
    # if not result_3.any():
    #   cat_image = cat_image
    # else:
    #   cat_image = np.concatenate([cat_image, result_3], axis = 1)

    cv2.imwrite(filepath, cat_image * 255.0)
    print(filepath)
    # im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    # im.save(filepath, 'png')


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    start_epoch = 0
    epoch = 9999999
    everyepoch = 3000
    start_step = 0
    numBatch = 1  # batchsize 10
    batch_size = 4
    decim_batch_size = 1
    adj_batch_size = 485
    patch_size = 384

    learning_rate = 0.0001
    sample_dir = './eval_restore/1'
    model_decom_CKPT_PATH = './check/MyNet_2000_best.pkl'
    sample_decom_dir = './result/decom_eval'
    checkpoints = torch.load(model_decom_CKPT_PATH)
    checkpoint = checkpoints['state_dict']

    decomposed_low_r_data_480 = []
    decomposed_low_i_data_480 = []
    decomposed_high_r_data_480 = []
    ###
    # train_low_data = read_directory('/home/intern2/jay/project/pt_kind/dataset/our485/low')
    # train_high_data = read_directory('/home/intern2/jay/project/pt_kind/dataset/our485/low')

    R_low = []
    R_high = []
    I_low_3 = []
    I_high_3 = []

    output_R_low = R_low
    output_R_high = R_high
    output_I_low = I_low_3
    output_I_high = I_high_3

    model_restore = RestorationNet()
    model_restore = nn.DataParallel(model_restore).cuda()
    optimizer = optim.Adam(model_restore.parameters(), lr=learning_rate)
    ##dloss = decom_loss()
    # dloss.cuda()

    ##dataloader
    dealDataset = DealDataset()

    train_loader = DataLoader(dataset=dealDataset,
                              batch_size=decim_batch_size,
                              shuffle=True)  # ,num_workers=16
    # batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")  #
    # batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")  #

    eval_imgs = read_directory('/home/intern2/jay/project/pt_kind/dataset/eval15/low')
    eval_img = load_images(cv2.imread('/home/intern2/jay/project/pt_kind/dataset/eval15/low/1.png'))

    ##   decom model
    model_decom = DecomNet()
    model_decom = nn.DataParallel(model_decom).cuda()
    model_decom.load_state_dict(checkpoint)
    model_decom.eval()

    ## run decom_model ------------- why use RR2   preprocess
    for i, data in enumerate(train_loader):
        print('i :', i)
        train_low, train_high = data
        
        train_low = train_low.permute([0, 3, 1, 2]).cuda()
        train_high = train_high.permute([0, 3, 1, 2]).cuda()
        
        R_low, I_low = model_decom(train_low)
        R_high, I_high = model_decom(train_high)

        R_high = torch.pow(R_high, 1.2)
        # R_low = torch.squeeze(R_low)
        # I_low = torch.squeeze(I_low)

        I_low = I_low.permute([0, 2, 3, 1]).squeeze(0).cpu().detach().numpy()
        R_low = R_low.permute([0, 2, 3, 1]).squeeze(0).cpu().detach().numpy()
        R_high = R_high.permute([0, 2, 3, 1]).squeeze(0).cpu().detach().numpy()

        # R_low = np.squeeze(R_low)
        # I_low = np.squeeze(I_low)
        decomposed_low_i_data_480.append(I_low)  # I_low
        decomposed_low_r_data_480.append(R_low)

        #print('R_i_shape', I_low.shape)
        decomposed_high_r_data_480.append(R_high)
    # print('decomposed_low_i_data_480 shape  ',decomposed_low_i_data_480.shape)
    decomposed_low_i_data_480 = np.array(decomposed_low_i_data_480)
    decomposed_low_r_data_480 = np.array(decomposed_low_r_data_480)
    decomposed_high_r_data_480 = np.array(decomposed_high_r_data_480)
    # print('decomposed_low_i_data_480 shape  ',decomposed_low_i_data_480.shape)

    # print('shape ',decomposed_low_i_data_480.shape)

    adjDataset = adjustdataset(decomposed_low_i_data_480, decomposed_low_r_data_480, decomposed_high_r_data_480)
    

    train_loader = DataLoader(dataset=adjDataset,
                              batch_size=adj_batch_size,
                              shuffle=True)  # ,num_workers=16
    reloss =  restore_loss().cuda()
    start = time.time()
    for epo in range(epoch):
        model_restore.train()  #
        print('epoch  :', epo)
        initial_lr = learning_rate

        if epo<=800:
            lr = initial_lr
        elif epo<=1250:
            lr = initial_lr/2
        elif epo<=1500:
            lr = initial_lr/4 
        else:
            lr = initial_lr/10 
        optimizer.param_groups[0]['lr'] = lr
        for i, data in enumerate(train_loader):
            # for i, in range(485):
            #  train_loader
            # data = torch.Tensor(data)

            train_r, train_i, train_hr = data
            #print('shape :',train_r.shape)
            train_r = train_r.permute([0, 3, 1, 2])#.cuda()
            train_i = train_i.permute([0, 3, 1, 2])#.cuda()
            train_hr = train_hr.permute([0, 3, 1, 2])#.cuda()
            j = train_r.shape[0]
            m = j // batch_size

            if (m * 10 < j):
                m = m + 1

            for index in range(m):
                #print('index ', index)
                if ((index + 1) * batch_size > train_r.shape[0]):
                    t = (index + 1) * batch_size - j
                else:
                    t = batch_size
                #print('train _ r  shape :',train_r[index * batch_size:index * batch_size + t, :, :, :].shape)
                #print('train _ i  shape :',train_i[index * batch_size:index * batch_size + t, :, :, :].shape)
                R_low_r = model_restore(train_r[index * batch_size:index * batch_size + t, :, :, :],
                                        train_i[index * batch_size:index * batch_size + t, :, :, :])
                #model_restore(train_r[index * batch_size:index * batch_size + t, :, :, :],
                 #                       train_i[index * batch_size:index * batch_size + t, :, :, :])
                print('R_low shape :',R_low_r.shape)
                #print('train_hr shape :', train_hr[index * 10:index * 10 + t, :, :, :].shape)
                loss = reloss(R_low_r,train_hr[index * batch_size:index * batch_size + t, :, :, :])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('loss :',loss)
                
                # R_low_i = model_restore( train_i[index * 10:index * 10 + t, :, :, :])
                # R_high_r = model_restore( train_hr[index * 10:index * 10 + t, :, :, :])
                # print('train pos ',train_low[index * 10:index * 10 +t,:,:,:].device)
                # print('w:',index * 10)
                # loss = dloss(train_low[index * 10:index * 10 + t, :, :, :],
                #              train_high[index * 10:index * 10 + t, :, :, :], R_low, I_low, R_high, I_high).cuda()
                # # print('2 ---------- ')
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                # print('loss :', loss)
                #
                if ((epo + 1) % everyepoch == 0):
                     img = np.array(eval_img)
                     img = torch.FloatTensor(img)  # .unsqueeze(0)
                     #print('shape 1',img.shape)
                     img = img.unsqueeze(0)
                     # print('shape 2',img[0].shape)
                     # img = img.unsqueeze(0)
                     # print('shape 3',img[(epo+1)//200].shape)
                     img = img.permute([0, 3, 1, 2]).cuda()
                     print('shape 2',img.shape)
                     dr,di = model_decom(img)
                     print('dr  ',dr.shape)
                     print('di  ',di.shape)
                     R_high = model_restore(train_r[index * batch_size:index * batch_size + t, :, :, :],train_i[index * batch_size:index * batch_size + t, :, :, :])
                     print('r_high  ',R_high.shape)
                     #I_high = torch.cat([I_high, I_high, I_high], axis=1)
                     R_high = R_high.permute([0, 2, 3, 1])
                     dr = dr.permute([0, 2, 3, 1])
                     #I_high = I_high.permute([0, 2, 3, 1])
                     print('dr shape',dr.shape)
                     print('R_high',R_high.shape)
                     save_images(os.path.join(sample_dir, 'low_%d.png' % (epo)), dr,R_high)
    end = time.time()
    print('time :',end -start)
    torch.save({'state_dict': model_restore.state_dict(), 'epoch': epoch}, 'MyNet_retore' + str(epoch) + '_best.pkl')
    print('Save best statistics done!')


