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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


#########################  decomnet


#########################




def load_images(im):
    img = np.array(im, dtype="float32") / 255.0
    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = np.float32((img - img_min) / np.maximum((img_max - img_min), 0.001))  #
    return img_norm


def pt_data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        # return np.flipud(image) torch.flip()
        return torch.flip(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        # return np.rot90(image)
        return torch.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        # image = np.rot90(image)
        # return np.flipud(image)
        image = torch.rot90(image)
        return torch.flip(image)
    elif mode == 4:
        # rotate 180 degree
        # return np.rot90(image, k=2)
        return torch.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        # image = np.rot90(image, k=2)
        # return np.flipud(image)
        image = torch.rot90(image, k=2)
        return torch.flip(image)
    elif mode == 6:
        # rotate 270 degree
        # return np.rot90(image, k=3)
        return torch.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        # image = np.rot90(image, k=3)
        # return np.flipud(image)
        image = torch.rot90(image, k=3)
        return torch.flip(image)


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
    def __init__(self, low_i, high_i):
        self.low_i = low_i
        self.high_i = high_i
        self.len = len(high_i)

    def __getitem__(self, index):
        # self.low_img[index], self.high_img[index]
        # print('low_i shape', self.low_i.shape)

        # 1
        i_low_expand = self.low_i[index]
        i_high_expand = self.high_i[index]
        # print('i_low_expand shape',i_low_expand.shape)  # 1 400 600

        # 2
        h = self.low_i[index].shape[0]
        w = self.low_i[index].shape[1]
        x = random.randint(0, h - patch_size)  #
        y = random.randint(0, w - patch_size)  #
        i_low_data_crop = i_low_expand[x: x + patch_size, y: y + patch_size, :]
        i_high_data_crop = i_high_expand[x: x + patch_size, y: y + patch_size, :]  # 1 385 385
        # print('i_low_data_crop shape',i_low_data_crop.shape)

        # i_low_data_crop = i_low_data_crop.permute([1,2,0])
        # i_high_data_crop = i_high_data_crop.permute([1,2,0])
        # 3
        rand_mode = random.randint(0, 7)  #
        # print('i_high_data_crop change :',i_high_data_crop.shape)
        patch_input_low_i = data_augmentation(
            i_low_data_crop, rand_mode)
        patch_input_high_i = data_augmentation(
            i_high_data_crop, rand_mode)  #

        # print('patch_input_high_i shape',patch_input_high_i.shape)

        # 4 ratio
        i_low_data_crop = i_low_data_crop.copy()
        i_high_data_crop = i_high_data_crop.copy()
        patch_input_low_i = patch_input_low_i.copy()
        patch_input_high_i = patch_input_high_i.copy()

        i_low_data_crop = torch.Tensor(i_low_data_crop).permute([2, 0, 1])
        i_high_data_crop = torch.Tensor(i_high_data_crop).permute([2, 0, 1])
        patch_input_low_i = torch.Tensor(patch_input_low_i).permute([2, 0, 1]).cuda()
        patch_input_high_i = torch.Tensor(patch_input_high_i).permute([2, 0, 1]).cuda()
        # print('last patch_input_low_i',patch_input_low_i.shape)

        ratio = torch.mean(i_low_data_crop / (i_high_data_crop + 0.0001))
        print('ratio :', ratio)
        # 5
        i_low_data_ratio = torch.ones(patch_size, patch_size) * (1 / ratio + 0.0001)
        i_low_ratio_expand = i_low_data_ratio.unsqueeze(0).cuda()

        i_high_data_ratio = torch.ones(patch_size, patch_size) * (ratio)
        i_high_ratio_expand = i_high_data_ratio.unsqueeze(0).cuda()

        # print('i_high_ratio_expand',i_high_ratio_expand.shape)# 1 384 384

        # 6

        rand_mode = np.random.randint(0, 2)
        if rand_mode == 1:
            return patch_input_low_i, patch_input_high_i, i_low_ratio_expand, i_high_ratio_expand
        else :
            return patch_input_high_i, patch_input_low_i, i_high_ratio_expand, i_low_ratio_expand

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

    # low_img=read_directory('/home/intern2/jay/project/pt_kind/dataset/our485/low')


# high_img = read_directory('/home/intern2/jay/project/pt_kind/dataset/our485/high')

# low_img = torch.tensor(low_img[0:100,:,:,:]).cuda()
# high_img = torch.tensor(high_img).cuda()

def grad_loss2(input_i_low, input_i_high):
    x_loss = torch.pow((torch.sub(gradient2(input_i_low, 'x'), gradient2(input_i_high, 'x'))), 2)
    y_loss = torch.pow((torch.sub(gradient2(input_i_low, 'y'), gradient2(input_i_high, 'y'))), 2)
    grad_loss_all = torch.mean(x_loss + y_loss)
    return grad_loss_all


class adjust_loss(nn.Module):
    def __init__(self):
        super(adjust_loss, self).__init__()

    def forward(self, output_i, input_high_i):
        loss_grad = grad_loss2(output_i, input_high_i)
        loss_square = torch.mean(
            torch.pow(torch.sub(output_i, input_high_i), 2))  # * ( 1 - input_low_r ))#* (1- input_low_i)))
        loss_adjust = torch.add(loss_square, loss_grad)
        return loss_adjust


def save_images2(filepath, result_1, result_2):
    result_2 = result_2.cpu().detach().numpy()
    # print('result1 shape',result_1.shape)
    # print('result2 shape',result_2.shape)
    # result_1 = np.squeeze(result_1)
    # result_2 = np.squeeze(result_2)
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
    epoch = 2000
    everyepoch = 10
    start_step = 0
    numBatch = 1  # batchsize 10
    batch_size = 10
    decim_batch_size = 1
    adj_batch_size = 485
    patch_size = 48

    learning_rate = 0.0001
    sample_dir = './eval_restore/1'
    sample_dir = './eval_adjust/1'
    model_decom_CKPT_PATH = 'MyNet_2000_best.pkl'
    sample_decom_dir = './result/decom_eval'
    checkpoints = torch.load(model_decom_CKPT_PATH)
    checkpoint = checkpoints['state_dict']

    decomposed_low_r_data_480 = []
    decomposed_low_i_data_480 = []
    decomposed_high_r_data_480 = []
    decomposed_high_i_data_480 = []
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

    model_adjust = AdjustNet()
    model_adjust = nn.DataParallel(model_adjust).cuda()
    optimizer = optim.Adam(model_adjust.parameters(), lr=learning_rate)
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

    start = time.time()

    for i, data in enumerate(train_loader):
        # print('i :', i)
        train_low, train_high = data

        train_low = train_low.permute([0, 3, 1, 2]).cuda()
        train_high = train_high.permute([0, 3, 1, 2]).cuda()

        R_low, I_low = model_decom(train_low)
        R_high, I_high = model_decom(train_high)

        I_low = I_low.permute([0, 2, 3, 1]).squeeze(0).cpu().detach().numpy()  # .permute([0, 2, 3, 1])
        I_high = I_high.permute([0, 2, 3, 1]).squeeze(0).cpu().detach().numpy()  # .permute([0, 2, 3, 1])

        #
        decomposed_low_i_data_480.append(I_low)  # I_low
        decomposed_high_i_data_480.append(I_high)

    collect_low_i = decomposed_low_i_data_480[0:450]
    collect_high_i = decomposed_high_i_data_480[0:450]

    eval_imgs_low_i = decomposed_low_i_data_480[451:480]
    eval_imgs_high_i = decomposed_high_i_data_480[451:480]

    adjDataset = adjustdataset(collect_low_i, collect_high_i)

    train_loader = DataLoader(dataset=adjDataset,
                              batch_size=adj_batch_size,
                              shuffle=True)  # ,num_workers=16
    reloss = adjust_loss().cuda()
    for epo in range(epoch):
        model_adjust.train()  #
        # print('epoch  :', epo)
        for i, data in enumerate(train_loader):
            patch_input_low_i, patch_input_high_i, i_low_ratio_expand, i_high_ratio_expand = data
            j = patch_input_low_i.shape[0]
            m = j // batch_size
            if (m * 10 < j):
                m = m + 1
            for index in range(m):
                rand_mode = np.random.randint(0, 2)
                if ((index + 1) * batch_size > patch_input_low_i.shape[0]):
                    t = (index + 1) * batch_size - j
                else:
                    t = batch_size
                # if rand_mode == 1:
                #    lowi = model_adjust(patch_input_low_i[index * batch_size:index * batch_size + t, :, :, :],
                #                        i_low_ratio_expand[index * batch_size:index * batch_size + t, :, :, :])
                #    rand = patch_input_high_i[index * batch_size:index * batch_size + t, :, :, :]
                #else:
                #    rand = patch_input_low_i[index * batch_size:index * batch_size + t, :, :, :]
                #    lowi = model_adjust(patch_input_high_i[index * batch_size:index * batch_size + t, :, :, :],
                #                        i_high_ratio_expand[index * batch_size:index * batch_size + t, :, :, :])
                
                lowi = model_adjust(patch_input_low_i[index * batch_size:index * batch_size + t, :, :, :],
                                        i_low_ratio_expand[index * batch_size:index * batch_size + t, :, :, :])
                print('data shape,', patch_input_low_i.shape)
                loss = reloss(lowi, patch_input_high_i[index * batch_size:index * batch_size + t, :, :, :])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('epoch:%d loss:%4.4f' % (epo, loss))
        if ((epo + 1) % everyepoch == 0):
            # img = np.array(eval_img)
            # img = torch.FloatTensor(img)  # .unsqueeze(0)
            # print('shape 1',img.shape)
            # img = img.unsqueeze(0)
            # print('shape 2',img[0].shape)
            # img = img.unsqueeze(0)
            # print('shape 3',img[(epo+1)//200].shape)
            s_low = eval_imgs_low_i[0]
            img = torch.Tensor(s_low).permute([2, 0, 1]).unsqueeze(0).cuda()
            rand_ratio = np.random.random(1) * 2
            h = img.shape[2]
            w = img.shape[3]
            input_uu_i_ratio = torch.Tensor(np.ones([h, w]) * rand_ratio).unsqueeze(0).unsqueeze(0).cuda()
            print('shape 2', img.shape, input_uu_i_ratio.shape)
            i_low = model_adjust(img, input_uu_i_ratio)
            i_low = i_low.squeeze(0).permute([1, 2, 0])
            print('ilow_shape  ', i_low.shape)
            save_images2(os.path.join(sample_dir, 'low_%d_%5f.png' % (epo,rand_ratio)), s_low,i_low)
    torch.save({'state_dict': model_adjust.state_dict(), 'epoch': epoch},'MyNet_adjust' + str(epoch) + '_best.pkl')
    print('Save best statistics done!')