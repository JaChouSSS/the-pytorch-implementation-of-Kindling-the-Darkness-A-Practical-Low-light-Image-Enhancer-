from model3 import *
import os
from pt_decomposition import  *
import cv2
from glob import glob



def read_directory2(directory_name):
    array_of_img = []
    name = []
    #print(os.listdir(directory_name))
    for filename in os.listdir(directory_name):
        
        img = cv2.imread( directory_name + "/" +filename)#directory_name + "/" +
        img = cv2.resize(img,(1024,1024))
        img = load_images(img)
        array_of_img.append(img)
        name.append(filename)
        
        print(filename)
        #print('img:',img)
    return array_of_img,name
    
    
    
class testDataset(Dataset):
    """
    """

    def __init__(self):

        #self.low_img=read_directory('./dataset/eval15/low')
        #self.high_img = read_directory('./dataset/eval15/high')
        #self.evl = read_directory('/home/share/data/Relighting/track1/track1_validation/input/')
        self.evl,self.name = read_directory2('/home/intern2/jay/project/pt_kind/dataset/our485/low')#here/home/intern2/jay/pro2/result/outa/
        #/home/intern2/jay/project/KinD-master/track1/track1_validation/input/

        self.len = len(self.evl)
    def __getitem__(self, index):

        return self.evl[index],self.name[index]#, self.high_img[index]
    def __len__(self):
        return self.len


def save_images(filepath,filepath2, result_1, result_2 = None, result_3 = None):
    result_2 = torch.cat([result_2, result_2, result_2], axis=3)
    result_1 = result_1.cpu().detach().numpy()
    result_2 = result_2.cpu().detach().numpy()
    print('result1 shape',result_1.shape)
    print('result2 shape',result_2.shape)
    
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)
    #result_3 = np.squeeze(result_3)

    #if not result_2.any():
    #    cat_image = result_1
    #else:
    cat_image = np.concatenate([result_1, result_2], axis = 1)
    #if not result_3.any():
     #   cat_image = cat_image
    #else:
     #   cat_image = np.concatenate([cat_image, result_3], axis = 1)
    print('cat_image :',cat_image.shape)
    cv2.imwrite(filepath,cat_image * 255.0)
    #cv2.imwrite(filepath2,result_2 * 255.0)
    #im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    #im.save(filepath, 'png')
    
def save_images2(filepath, result_1, result_2):
    #result_2 = torch.cat([result_2, result_2, result_2], axis=3)
    result_1 = result_1.cpu().detach().numpy()
    result_2 = result_2.cpu().detach().numpy()
    print('result1 shape',result_1.shape)
    print('result2 shape',result_2.shape)
    
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)
    #result_3 = np.squeeze(result_3)

    #if not result_2.any():
    #    cat_image = result_1
    #else:
    cat_image = np.concatenate([result_1, result_2], axis = 1)
    print('cat_image :',cat_image.shape)
    #if not result_3.any():
     #   cat_image = cat_image
    #else:
     #   cat_image = np.concatenate([cat_image, result_3], axis = 1)
    print('file path:',filepath)
    cv2.imwrite(filepath,result_1 * 255.0)
    
    #im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    #im.save(filepath, 'png')


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    CKPT_PATH = 'MyNet_2000_re_best.pkl'
    CKPT_PATH2 = 'MyNet_retore1000_best.pkl'
    CKPT_PATH3 = 'MyNet_adjust2000_best.pkl'
    #sample_dir = '/home/intern2/jay/project/KinD-master/zz_result_r2/'#/home/intern2/jay/project/AboutStyleTrans/pix2pix-pytorch-master2222222/dataset/track1/train/a/
    sample_dir = './test_decom'
    sample_dir2 = './result/restore_eval/1'
    sample_dir3 = './result/adjust_eval/1'
    sample_dir4 = './result/result_eval/1'
    checkpoints = torch.load(CKPT_PATH)
    checkpoint = checkpoints['state_dict']
    step = checkpoints['epoch']
    #/home/intern2/jay/project/KinD-master/zz_result_r2/

    model = DecomNet()
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint)
    model.eval()
    
    model_restore = RestorationNet()
    model_restore = nn.DataParallel(model_restore).cuda()
    model_restore.load_state_dict(torch.load(CKPT_PATH2)['state_dict'])
    model_restore.eval()
    
    model_adjust = AdjustNet()
    model_adjust = nn.DataParallel(model_adjust).cuda()
    model_adjust.load_state_dict(torch.load(CKPT_PATH3)['state_dict'])
    model_adjust.eval()
    
    
    

    testset= testDataset()
    loader = DataLoader(dataset=testset,
                              batch_size=1,
                              shuffle=False)
                              
    eval_img = load_images(cv2.imread('/home/intern2/jay/project/pt_kind/Image001.png'))
    
    #eval_low_data_name = glob('/home/intern2/jay/project/KinD-master/track1/track1_validation/input/*')
    eval_low_data_name = os.listdir('/home/intern2/jay/project/pt_kind/dataset/our485/low')#
    #/home/intern2/jay/project/KinD-master/track1/track1_validation/input/
    print('name ',eval_low_data_name)
    
    for i, data in enumerate(loader):
    
        evl_imgs,name= data
        
        print('data name ',name[0])
        print('aaaaaaaaaaaaaaa shape', evl_imgs.shape)
        #batch_input_low = torch.Tensor(low).permute([0,3,1,2]).cuda()
        #high = torch.Tensor(high).permute([0,3,1,2]).cuda()
        low = torch.Tensor(evl_imgs).permute([0,3,1,2]).cuda()
        h = low.shape[2]
        w = low.shape[3]
        
        ###### decom
        #h_R,h_l = model(high)
        low_R,low_l2 = model(low)
        
        # print('eval_img shape',eval_img.shape)#1024 1024 3
        # print('eval_img shape',eval_img.shape)
        #eval_img = torch.Tensor(eval_img).permute(2,0,1).unsqueeze(0).cuda()
        elow_R,elow_l = model(low)
        
        #cc = eh_l
        #h_l = torch.cat([eh_l, eh_l, eh_l], axis=1)
        #eh_R=eh_R.permute([0,2,3,1])
        #eh_l=eh_l.permute([0,2,3,1])
        elow_R=elow_R.permute([0,2,3,1])
        elow_l=elow_l.permute([0,2,3,1])
        #cc = cc.permute([0,2,3,1])
        #cc = cc.cpu().detach().numpy()
        #print('h_l :',cc)
        #cv2.imwrite('./'+str(i)+'.png',cc[0] * 255.0)
        #print('h_l shape',h_l)
        print('imaga name ',eval_low_data_name[i])
        if(os.path.isdir(sample_dir) is False):
            os.mkdir(sample_dir)

        save_images(os.path.join(sample_dir,name[0]),os.path.join(sample_dir, 'ill_%d.png' % ( i+1)),elow_R,elow_l)
          
        
        
        
        
        
        
        # low_l = low_l2
        # print('low_R0 ',low_R.shape)# 1 , 3, 400,600
        # low_R = low_R.squeeze(0).permute(1,2,0).cpu().detach().numpy()
        # print('low_R1 ',low_R.shape)# 400 600 3
        # low_R = cv2.resize(low_R,(400,400))
        # print('low_R 2',low_R.shape)
        # low_l = cv2.resize(low_l.squeeze(0).permute(1,2,0).cpu().detach().numpy(),(400,400))
        # print('low_R3 ',low_R.shape)#400 400 3
        # print('low_l3 ',low_l.shape)# 400 400
        # low_R = torch.Tensor(low_R).permute(2,0,1).cuda()
        # low_l = torch.Tensor(low_l).unsqueeze(2).permute(2,0,1).cuda()
        # print('low_R4 ',low_R.shape)
        # print('low_l4 ',low_l.shape)
        # #low_l = torch.cat([low_l,low_l,low_l],axis=0)
        # print('low_l5 ',low_l.shape)
        
        
        #low_l2 = torch.cat([low_l2,low_l2,low_l2],axis=1)
        #print('low_l2 shape',low_l2.shape)
        #print('low_R shape',low_R.shape)
        ### restore low
        re_r = model_restore(low_R,low_l2)
        #print('re_r ',re_r.shape)
        #save_images2(os.path.join(sample_dir2, 'low_%d.png'%(i)),low_R.squeeze(0).permute([1,2,0]) , re_r.squeeze(0).permute([1,2,0]))
        
        #print(low_l2.shape,h_l.shape)
        #low_l = low_l.unsqueeze(0)
        ratio = 1.7#torch.mean(low_l2 / (h_l + 0.0001))#  high / low
        i_low_data_ratio = torch.ones(h, w) * (1 / ratio + 0.0001)
        i_low_ratio_expand = i_low_data_ratio.unsqueeze(0).cuda()
        
        ###adjust low
        a_l = model_adjust(low_l2,i_low_ratio_expand)
        #a_l = a_l.squeeze(0).permute([1, 2, 0])
        #print('ilow_shape  ', a_l.shape,low_l2.shape)# 1, 1, 400, 600
        
        #save_images2(os.path.join(sample_dir3, 'low_%5f.png' % (ratio)), low_l2.squeeze(0).permute([2,0,1]), a_l.squeeze(0).permute([2,0,1]))
        #print('=============================')
        #print('low shape',low.shape)
        #print('h_R',h_R.shape)
        #print('re_r shape',re_r.shape)
        ##print('h_l shape',h_l.shape) 
        #print('a_l shape',a_l.shape)    
        #print('=============================')      
        #re_r shape torch.Size([1, 3, 400, 400])
        #re_r = cv2.resize(re_r.squeeze(0).permute(1,2,0).cpu().detach().numpy(),(400,400))
        #print('==========',re_r.shape)#400 600 ,3
        #re_r = torch.Tensor(re_r).permute(2,0,1).unsqueeze(0).cuda()
        
        #low.cuda()
        #re_r.cuda()
        #low_r#
        #a_l.cuda()
        #low_l2
        
        #fusion = PostProcess(low.cuda(),h_R.cuda(),re_r.cuda(),h_l.cuda(),a_l.cuda())
        #print('low shape',low.shape)
        #print('re_r shape',re_r.shape)
        #print('low_R shape',low_R.shape)
        #print('a_l shape',a_l.shape)    
        #print('low_l2 shape',low_l2.shape) 
        
        fusion = PostProcess(low.cuda(),re_r.cuda(),low_R.cuda(),a_l.cuda(),low_l2.cuda())
        
        #print('fusion shape',fusion.shape)
        #print('low 3232323',low.squeeze(0).shape)
        #print('low ',low.squeeze(0).permute([2,0,1]).shape)
        sample_dir4 = './final_result'
        if(os.path.isdir(sample_dir4) is False):
            os.mkdir(sample_dir4)
        save_images2(os.path.join(sample_dir4, 'low_%d.png' % (i)), low.squeeze(0).permute([1,2,0]), fusion.squeeze(0).permute([1,2,0]))
        
        #cc = h_l
        #h_l = torch.cat([h_l, h_l, h_l], axis=1)
        #h_R=h_R.permute([0,2,3,1])
        #h_l=h_l.permute([0,2,3,1])
        #low_R=low_R.permute([0,2,3,1])
        #low_l=low_l.permute([0,2,3,1])
        #cc = cc.permute([0,2,3,1])
        #cc = cc.cpu().detach().numpy()
        #print('h_l :',cc)
        #cv2.imwrite('./'+str(i)+'.png',cc[0] * 255.0)
        #print('h_l shape',h_l)
        #save_images(os.path.join(sample_dir, 'low_%d.png' % ( i+1)),low_R,low_l)
        #print('eval_img shape',eval_img.shape)
if __name__ == "__main__":
  main()
