import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from torch.autograd import Variable  
from collections import OrderedDict
import pdb


class DecomNet(nn.Module):
    def __init__(self):
        super(DecomNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.conv2_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.conv3_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.deconv4_1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deconv5_1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv5_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.conv3_2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        conv1 = self.conv1(x)
        r_conv2 = self.conv2_1(conv1)
        r_conv3 = self.conv3_1(r_conv2)
        # print("r_conv2 shape: ", r_conv2.shape)
        # print("r_conv3 shape: ", r_conv3.shape)
        # print("deconv of r_conv3 shape: ", self.deconv4_1(r_conv3).shape)
        r_conv4 = self.conv4_1(torch.cat((r_conv2, self.deconv4_1(r_conv3)), 1))
        # print("conv1 shape: ", conv1.shape)
        # print("r_conv4 shape: ", r_conv4.shape)
        # print("deconv of r_conv4 shape: ", self.deconv5_1(r_conv4).shape)
        r_conv5 = self.conv5_1(torch.cat((conv1, self.deconv5_1(r_conv4)), 1))
        ReflectOut = self.conv6_1(r_conv5)

        i_conv2 = self.conv2_2(conv1)
        IllumiOut = self.conv3_2(torch.cat((i_conv2, r_conv5), 1))

        return ReflectOut, IllumiOut


class RestorationNet(nn.Module):
    def __init__(self):
        super(RestorationNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.deconv6 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.deconv7 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.deconv8 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2)
        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.deconv9 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.conv9 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, input_r, input_i):
        input_all = torch.cat((input_r, input_i), 1)
        conv1 = self.conv1(input_all)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        print("conv4 shape: ", conv4.shape)
        print("deconv6 shape: ", self.deconv6(conv5).shape)
        conv6 = self.conv6(torch.cat((conv4, self.deconv6(conv5)), 1))
        conv7 = self.conv7(torch.cat((conv3, self.deconv7(conv6)), 1))
        conv8 = self.conv8(torch.cat((conv2, self.deconv8(conv7)), 1))
        conv9 = self.conv9(torch.cat((conv1, self.deconv9(conv8)), 1))
        return self.conv10(conv9)


class AdjustNet(nn.Module):
    def __init__(self):
        super(AdjustNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, input_ratio):
        # ratio = input_ratio.expand(x.shape)
        input_all = torch.cat((x, input_ratio), 1)
        return self.conv4(self.conv3(self.conv2(self.conv1(input_all))))


class GaussianBlur(nn.Module):
    def __init__(self):
        super(GaussianBlur, self).__init__()
        _kernel = [[0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902],
                  [0.01330621, 0.0596343,  0.09832033, 0.0596343,  0.01330621],
                  [0.02193823, 0.09832033, 0.16210282, 0.09832033, 0.02193823],
                  [0.01330621, 0.0596343,  0.09832033, 0.0596343,  0.01330621],
                  [0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902]]        

        _kernel = torch.tensor(_kernel).view(1, 1, 5, 5)
        # print("Kernel shape: ", kernel.shape)
        self.weight = _kernel #nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=2)


class RGB2Gray(nn.Module):
    def __init__(self):
        super(RGB2Gray, self).__init__()
        _kernel = [0.2125, 0.7154, 0.0721]
        _kernel = torch.tensor(_kernel).view(1, 3, 1, 1)
        self.weight = _kernel 

    def forward(self, x):
        gray =  F.conv2d(x, self.weight)
        return gray

def torch_min(x):
    relu = nn.ReLU()
    x = (x*2)**0.5
    x = -1.0 * relu(-1.0 * x + 1)
    x = x + 1
    return x

def PostProcess(input, reflec_high, reflect_low, illumi_high, illumi_low):
    rgb2gray = RGB2Gray()
    gaussian = GaussianBlur()
    low_i = gaussian(rgb2gray(reflect_low))
    # low_i = torch.min((low_i*2) ** 0.5, torch.tensor(1.0))
    low_i = torch_min(low_i)
    low_i = torch.cat((low_i, low_i, low_i), 1)
    result_denoise = reflec_high * low_i
    illumi_high = torch.cat((illumi_high, illumi_high, illumi_high), 1)
    illumi_low = torch.cat((illumi_low, illumi_low, illumi_low), 1)
    fusion4 = result_denoise * illumi_high
    # fusion2 = illumi_low * input + (1-illumi_low))*fusion4
    fusion2 = illumi_low * input + (-1.0*illumi_low + 1) * fusion4

    return fusion2

def process_pkl(pkl_name):
    state_dict = torch.load(pkl_name)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.'
        new_state_dict[name] = v

    return new_state_dict


class IllumiEnhanceNet(nn.Module):
    def __init__(self, subnet):
        super(IllumiEnhanceNet, self).__init__()
        self.decom = DecomNet()
        self.decom.load_state_dict(process_pkl(subnet[0]))
        self.restoration = RestorationNet()
        self.restoration.load_state_dict(process_pkl(subnet[1]))
        self.adjust = AdjustNet()
        self.adjust.load_state_dict(process_pkl(subnet[2]))

    def forward(self, x, ratio,):
        reflect_low, illumi_low = self.decom(x)
        reflect_high = self.restoration(reflect_low, illumi_low)
        illumi_high = self.adjust(illumi_low, ratio)
        result = PostProcess(x, reflect_high, reflect_low, illumi_high, illumi_low)
        return result

class MulTest(nn.Module):
    def __init__(self):
        super(MulTest, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = (x*2)**0.5
        x = -1.0 * self.relu(-1.0 * x + 1)
        x = x + 1
        return x



if __name__ == "__main__":
    decom_model = DecomNet()
    restoration_model = RestorationNet()
    adjust_model = AdjustNet()

    subnet = ["../MyNet_2000_best.pkl", "../MyNet_restore1000_best.pkl", "../MyNet_adjust2000_best.pkl"]

    x = torch.ones(1, 3, 480, 320)
    input_ratio = 5.0
    input_ratio = torch.tensor(input_ratio).expand(1, 1, 480, 320)
    '''
    reflect_low, illumi_low = decom_model(x)
    reflect_high = restoration_model(reflect_low, illumi_low)
    illumi_high = adjust_model(illumi_low, input_ratio)
    fusion2 = PostProcess(x, reflect_high, reflect_low, illumi_high, illumi_low)
    print("x shape: ", x.shape)
    print("relect_low shape: ", reflect_low.shape)
    print("illumi_low shape: ", illumi_low.shape)
    print("reflec_high shape: ", reflect_high.shape)
    print("illumi_high shape: ", illumi_high.shape)
    print("fusion2 shape: ", fusion2.shape)
    '''
    print("========")
    
    illumi_enhance = IllumiEnhanceNet(subnet)
    
    # torch.save(illumi_enhance.state_dict(), "../DarkEnhance.pkl")
    # illumi_enhance.load_state_dict(torch.load("../DarkEnhance.pkl"))

    illumi_enhance.eval()
    img = illumi_enhance(x, input_ratio) 
    img = img * 4096
    print(img[0,0,0,:])
    print(img[0,0,1,:])
    print(img[0,0,2,:])
    print("img shape: ", img.shape)
    torch.onnx.export(illumi_enhance, (x, input_ratio), "../DarkEnhance.onnx", input_names=["input", "ratio"], 
		    output_names=["img"])
    pass
